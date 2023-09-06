from fastapi import FastAPI, APIRouter
from src.api import people

router = APIRouter()
router.include_router(people.router)

app = FastAPI(title='Account', version='1.0.0')
app.include_router(router)