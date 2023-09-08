from fastapi import FastAPI, APIRouter
from src.api import people
from src.api import face
from src.api import recognition

router = APIRouter()
router.include_router(people.router)
router.include_router(face.router)
router.include_router(recognition.router)

app = FastAPI(title='Account', version='1.0.0')
app.include_router(router)