from fastapi import APIRouter, Response, status, UploadFile, File
from fastapi.responses import JSONResponse
from src.schemas.validation import ImageValidation
from src.pipeline.face_crud import FaceCRUD
from urllib.parse import unquote
import cv2
import numpy as np

face_crud = FaceCRUD()
router = APIRouter(prefix='/people', tags=['faces'])