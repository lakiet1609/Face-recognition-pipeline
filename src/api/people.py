from fastapi import APIRouter, Response, status
from urllib.parse import unquote
from src.pipeline.people_crud import PersonCRUD

person_crud = PersonCRUD()

router = APIRouter(prefix='/people', tags=['people'])

@router.get('', status_code=status.HTTP_200_OK)
async def get_all_people(skip: int, limit:int):
    people_list = person_crud.select_all_people(skip, limit)
    return people_list

@router.get('/{person_id}', status_code= status.HTTP_200_OK)
async def select_person_by_ID(person_id: str):
    person_id = unquote(person_id)
    person_doc = person_crud.select_person_by_id(person_id)
    return person_doc

@router.post('', status_code= status.HTTP_201_CREATED)
async def insert_one_person(person_id, name):
    person_id, name = unquote(person_id), unquote(name)
    person_doc = person_crud.insert_person(person_id, name)
    return person_doc

@router.post('/{person_id}/name', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def update_name(person_id: str, name: str):
    person_id, name = unquote(person_id), unquote(name)
    person_crud.update_person_name(person_id, name)

@router.delete('/{id}', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def delete_person_by_ID(id: str):
    person_crud.delete_person_by_id(id)

@router.delete('', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def delete_all_people():
    person_crud.delete_all_people()