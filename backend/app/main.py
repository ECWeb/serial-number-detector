from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.yolov8_ocr import detect_serial_number

import easyocr
import numpy as np
import cv2
from paddleocr import PaddleOCR

app = FastAPI()

easyocr_reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

paddleocr_reader = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False) 

# Allow Vue dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect_serial/image")
async def test_serial_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

   
    # easyocr test
    easyocr_results = easyocr_reader.readtext(image)
    easyocr_texts = [res[1] for res in easyocr_results]

    # paddleocr test
    paddleocr_results = paddleocr_reader.ocr(image, cls=False)
    paddleocr_texts = []
    for line in paddleocr_results:
        for box, (text, confidence) in line:
            paddleocr_texts.append(text)

    return JSONResponse(content={
        "easyocr": easyocr_texts,
        "paddleocr": paddleocr_texts
    })

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    serials = detect_serial_number(image_bytes)
    return {"serials": serials}



@app.get("/test")
def test():
    return {"status": "backend OK"}