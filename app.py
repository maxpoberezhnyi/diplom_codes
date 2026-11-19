import streamlit as st
import cv2
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import chromadb
from sentence_transformers import SentenceTransformer
import os

st.set_page_config(page_title="Відеоаналітика Edge AI", layout="wide")
st.title("🛡️ Інтелектуальна система відеоспостереження")

@st.cache_resource
def load_ai_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Моделі
    yolo = YOLO("yolov8n.pt")
    vlm_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    vlm_mod = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    
    # База даних
    client = chromadb.PersistentClient(path="./diploma_db")
    collection = client.get_or_create_collection(name="video_events")
    
    return device, yolo, vlm_proc, vlm_mod, embedder, collection

with st.spinner("Ініціалізація нейромереж (YOLO, BLIP, VectorDB)..."):
    device, yolo_detector, vlm_processor, vlm_model, embedder, collection = load_ai_models()

def calculate_iou(box1, box2):
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(b1_area + b2_area - intersection)

def generate_description(frame_rgb):
    image = Image.fromarray(frame_rgb)
    inputs = vlm_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = vlm_model.generate(**inputs, max_new_tokens=60)
    return vlm_processor.decode(out[0], skip_special_tokens=True)

tab1, tab2 = st.tabs(["📺 Аналіз відео (Ядро)", "🔍 Пошук подій (Інтерфейс)"])

with tab1:
    st.markdown("### 📥 1. Завантажте відео для аналізу")
    uploaded_file = st.file_uploader("Оберіть відеофайл (MP4, AVI)", type=['mp4', 'avi'])
    
    if uploaded_file:
        video_path = "current_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ Відео успішно завантажено!")
        
        iou_threshold = st.slider("Поріг чутливості руху (IoU):", 0.0, 1.0, 0.85, 0.05)
        
        if st.button("▶️ Запустити каскадний аналіз", type="primary"):
            col_video, col_log = st.columns([2, 1])
            
            video_placeholder = col_video.empty()
            log_placeholder = col_log.empty()
            event_logs = ["**Лог виявлених подій:**\n"]
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            frame_count = 0
            saved_events = 0
            previous_boxes = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                clean_frame = frame.copy()

                results = yolo_detector(frame, verbose=False, classes=[0, 2, 5, 7])
                current_boxes = results[0].boxes.xyxy.cpu().numpy()
                
                is_new_event = False
                if len(current_boxes) != len(previous_boxes):
                    is_new_event = True
                elif len(current_boxes) > 0:
                    iou_score = calculate_iou(current_boxes[0], previous_boxes[0])
                    if iou_score < iou_threshold:
                        is_new_event = True
                        
                for box in current_boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                if is_new_event:
                    timestamp = round(frame_count / fps, 2)

                    frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                    description = generate_description(frame_rgb)
                    
                    frame_vector = embedder.encode(description).tolist()
                    event_id = f"evt_{int(time.time())}_{saved_events}"

                    print(f"Збереження події... : {event_id}, Опис: {description}")

                    try:
                        collection.upsert(
                            ids=[event_id],
                            embeddings=[frame_vector],
                            documents=[description],
                            metadatas=[{"timestamp": timestamp}]
                        )
                        print(f"Збережено подію: {event_id}")
                    except Exception as e:
                        print(f"Помилка збереження: {e}")
                    
                    event_logs.append(f"⏱ **{timestamp}s**: {description}")
                    log_placeholder.info("\n\n".join(event_logs[-8:])) 
                    
                    previous_boxes = current_boxes
                    saved_events += 1
                
                frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb_display, channels="RGB", width="stretch")
                frame_count += 1
                
            cap.release()
            st.success(f"🎉 Аналіз завершено! Збережено подій: {saved_events}")

with tab2:
    st.markdown("### 🔎 2. Пошук подій у базі даних")
    query = st.text_input("Введіть запит (наприклад: 'car on the street'):", placeholder="Що шукаємо?")
    max_distance = st.slider("Максимальна відстань (менше = точніше):", 0.5, 2.0, 1.2, 0.1)

    if st.button("Знайти фрагмент", type="primary", key="search_btn"):
        if query:
            if not os.path.exists("current_video.mp4"):
                st.error("Спочатку завантажте відео на вкладці 'Аналіз відео'.")
            else:
                with st.spinner("Векторизація та пошук..."):
                    query_vector = embedder.encode(query).tolist()
                    results = collection.query(query_embeddings=[query_vector], n_results=5)
                    
                    valid_results = []
                    if results['distances'] and results['distances'][0]:
                        for i in range(len(results['distances'][0])):
                            dist = results['distances'][0][i]
                            if dist < max_distance:
                                valid_results.append({
                                    "distance": dist,
                                    "description": results['documents'][0][i],
                                    "timestamp": results['metadatas'][0][i]['timestamp']
                                })
                    
                    if valid_results:
                        st.success(f"✅ Знайдено {len(valid_results)} точних збігів!")
                        for res in valid_results:
                            with st.expander(f"⏱ Таймкод: {res['timestamp']} сек. | Відстань: {res['distance']:.2f} | Опис: {res['description']}"):
                                st.video("current_video.mp4", start_time=int(res['timestamp']))
                    else:
                        st.warning("🚷 За вашим запитом нічого не знайдено (результати перевищують поріг чутливості).")
        else:
            st.error("Будь ласка, введіть запит.")