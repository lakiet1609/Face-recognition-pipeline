from fastapi import APIRouter, Response, status
from urllib.parse import unquote
from src.pipeline.people_crud import PersonCRUD

person_crud = PersonCRUD()

router = APIRouter('/people', tags=['people'])

@router.get('', status=status.HTTP_200_OK)
async def get_all_people(skip: int, limit:int):
    people_list = person_crud.select_all_people(skip, limit)
    return people_list

@router.get('/{person_id}', status_code= status.HTTP_200_OK)
async def select_person_by_ID(person_id: str):
    person_id = unquote(person_id)
    person_doc = person_crud.select_person_by_id(person_id)
    return person_doc

