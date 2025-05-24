from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from app.utils.detector import detect_image
from PIL import Image
import io

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_with_boxes, labels = detect_image(image_bytes)

    # Картинка в байты
    img_io = io.BytesIO()
    Image.fromarray(image_with_boxes).save(img_io, format="JPEG")
    img_io.seek(0)

    # Просто возвращаем изображение без заголовков
    return StreamingResponse(img_io, media_type="image/jpeg")

# Выводим список обнаруженных знаков в JSON формате
@app.post("/detect/json/")
async def detect_json(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _, labels = detect_image(image_bytes)
    return JSONResponse({"detected": labels})
