from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from app.utils.detector import detect_image
from PIL import Image
import io

app = FastAPI()


@app.post("/detect/")
async def detect(file: UploadFile = File(...), confidence: float = Query(0.3, ge=0.0, le=1.0)):
    image_bytes = await file.read()
    image_with_boxes, labels = detect_image(image_bytes, confidence)

    img_io = io.BytesIO()
    Image.fromarray(image_with_boxes).save(img_io, format="JPEG")
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.post("/detect/json/")
async def detect_json(file: UploadFile = File(...), confidence: float = Query(0.3, ge=0.0, le=1.0)):
    image_bytes = await file.read()
    _, labels = detect_image(image_bytes, confidence)
    return JSONResponse({"detected": labels})
