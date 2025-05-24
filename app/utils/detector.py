from ultralytics import YOLO
from PIL import Image
import io
import json
import cv2
import numpy as np
from collections import Counter

# Загрузка модели и label map
model = YOLO("app/model/best.pt")
with open("app/model/human_label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

def detect_image(image_bytes, confidence=0.3, show_labels=True):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(img, save=False, conf=confidence, verbose=False)[0]

    detected = []
    counts = Counter()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = label_map.get(str(cls_id), f"Класс {cls_id}")
        conf = round(float(box.conf[0]), 2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detected.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
        counts[label] += 1

    if show_labels:
        img_with_boxes_bgr = results.plot()
    else:
        # Преобразуем PIL в BGR-формат для OpenCV
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Рисуем только прямоугольники (без текста)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        img_with_boxes_bgr = img_bgr

    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes_bgr, cv2.COLOR_BGR2RGB)

    return img_with_boxes_rgb, detected, dict(counts)
