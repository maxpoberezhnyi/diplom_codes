import cv2
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import chromadb
from sentence_transformers import SentenceTransformer

VIDEO_PATH = "test.mp4"
IOU_THRESHOLD = 0.85 
ACTIVE_RULE_TEXT = "людина впала на землю" 

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] Завантаження моделей...")
yolo_detector = YOLO("yolov8n.pt")

vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

chroma_client = chromadb.PersistentClient(path="./diploma_db")
collection = chroma_client.get_or_create_collection(name="video_events")
trigger_vector = embedder.encode(ACTIVE_RULE_TEXT).tolist()

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

def generate_semantic_description(frame_rgb):
    image = Image.fromarray(frame_rgb)
    inputs = vlm_processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = vlm_model.generate(**inputs, max_new_tokens=60)
        
    description = vlm_processor.decode(out[0], skip_special_tokens=True)
    return description

def process_video_stream():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frame_count = 0
    saved_events = 0
    previous_boxes = []
    
    print("\n[START] Аналіз відеопотоку...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = yolo_detector(frame, verbose=False, classes=[0, 2, 5, 7]) 
        current_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        is_new_event = False
        
        if len(current_boxes) != len(previous_boxes):
            is_new_event = True
        elif len(current_boxes) > 0:
            iou_score = calculate_iou(current_boxes[0], previous_boxes[0])
            if iou_score < IOU_THRESHOLD:
                is_new_event = True
                
        if is_new_event:
            timestamp = round(frame_count / fps, 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            t0 = time.perf_counter()
            description = generate_semantic_description(frame_rgb)
            vlm_time = (time.perf_counter() - t0) * 1000
            
            frame_vector = embedder.encode(description).tolist()
            collection.upsert(
                ids=[f"event_{saved_events}"],
                embeddings=[frame_vector],
                documents=[description],
                metadatas=[{"timestamp": timestamp}]
            )
            
            similarity = np.dot(frame_vector, trigger_vector) / (np.linalg.norm(frame_vector) * np.linalg.norm(trigger_vector))
            alert = "ТРИГЕР СПРАЦЮВАВ" if similarity > 0.65 else ""
            
            print(f"[{timestamp}s] Подія #{saved_events} | VLM: {vlm_time:.0f}ms | {description} {alert}")
            
            previous_boxes = current_boxes
            saved_events += 1
            
        frame_count += 1
        
    cap.release()
    print(f"\n[DONE] Оброблено кадрів відео: {frame_count}. Збережено семантичних подій: {saved_events}.")
    
    if frame_count > 0:
        compression = (1 - (saved_events / frame_count)) * 100
        print(f"Коефіцієнт компресії потоку (FRR): {compression:.2f}%")

if __name__ == "__main__":
    process_video_stream()