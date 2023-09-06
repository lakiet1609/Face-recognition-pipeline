from fastapi import APIRouter, Response, status, UploadFile, File
from fastapi.responses import JSONResponse
from src.schemas.validation import ImageValidation
from src.pipeline.face_crud import FaceCRUD
from urllib.parse import unquote
import cv2
import numpy as np

face_crud = FaceCRUD()
router = APIRouter(prefix='/people', tags=['faces'])

@router.get('/{person_id}/faces', status_code=status.HTTP_200_OK)
async def get_all_faces(person_id: str, skip: int, limit: int):
    face_docs = face_crud.select_all_face_of_person(person_id, skip, limit)
    return face_docs

@router.post('/{person_id}/faces', response_model=ImageValidation, status_code=status.HTTP_201_CREATED)
async def insert_one_face(person_id: str, image: UploadFile = File(...)):
    content = await image.read()
    image_buffer = np.frombuffer(content, np.uint8)
    img_decode = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    person_id = unquote(person_id)
    result = face_crud.insert_face(person_id, img_decode)
    return result

@router.delete('/{person_id}/faces', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def delete_one_face(person_id: str, face_id: str):
    face_crud.delete_face_by_ID(person_id, face_id)

@router.delete('/{person_id}/faces', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def delete_all_faces(person_id: str, face_id: str):
    face_crud.delete_all_face(person_id, face_id)