from ultralytics import YOLO
from PIL import Image
import io
import json
import cv2

# Загрузка модели и label map
model = YOLO("app/model/best.pt")
with open("app/model/human_label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

def detect_image(image_bytes, confidence=0.3):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Запуск модели с заданным порогом уверенности
    results = model.predict(img, save=False, conf=confidence, verbose=False)[0]

    # Рисуем боксы
    img_with_boxes_bgr = results.plot()
    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes_bgr, cv2.COLOR_BGR2RGB)

    detected = []
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

    return img_with_boxes_rgb, detected
