from fastapi import FastAPI, APIRouter
from src.api import people
from src.api import face

router = APIRouter()
router.include_router(people.router)
router.include_router(face.router)

app = FastAPI(title='Account', version='1.0.0')
app.include_router(router)