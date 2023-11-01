from fastapi import APIRouter, Response, status, UploadFile, File
from urllib.parse import unquote
import cv2
import numpy as np
from src.pipeline.face_recognition import FaceRecognition

face_recognition = FaceRecognition()

router = APIRouter(prefix='/recognition', tags=['prediction'])

@router.post("",status_code=status.HTTP_200_OK)
async def recognize(image: UploadFile = File(...)):
	content = await image.read()
	image_buffer = np.frombuffer(content, np.uint8)
	np_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
	recognition_result = face_recognition.get_recognize(np_image)
	return recognition_result

@router.post("/training",status_code=status.HTTP_200_OK)
async def train():
	face_recognition.get_reload()