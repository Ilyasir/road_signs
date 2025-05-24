from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import base64
from app.utils.detector import detect_image

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    confidence: float = Form(...),
    show_labels: bool = Form(False)
):
    image_bytes = await file.read()
    img_with_boxes, detected, counts = detect_image(image_bytes, confidence=confidence, show_labels=show_labels)

    img_io = io.BytesIO()
    Image.fromarray(img_with_boxes).save(img_io, format="JPEG")
    img_base64 = base64.b64encode(img_io.getvalue()).decode()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "image": img_base64,
        "labels": detected,
        "counts": counts,
        "confidence": confidence,
        "show_labels": show_labels
    })
