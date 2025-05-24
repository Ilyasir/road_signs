from ultralytics import YOLO
from PIL import Image
import io
import json
import cv2
import numpy as np

# Загрузка модели и label map
model = YOLO("app/model/best.pt")
with open("app/model/human_label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

def detect_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Запуск модели
    results = model.predict(img, save=False, conf=0.3, verbose=False)[0]

    # Рисуем боксы
    img_with_boxes_bgr = results.plot()
    # Преобразуем BGR в RGB для корректного отображения
    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes_bgr, cv2.COLOR_BGR2RGB) 

    # Собираем текстовый вывод
    detected = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = label_map.get(str(cls_id), f"Класс {cls_id}")
        conf = round(float(box.conf[0]), 2)
        detected.append(f"{label} ({conf})")

    return img_with_boxes_rgb, detected
