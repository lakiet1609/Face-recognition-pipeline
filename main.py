from src.pipeline.people_crud import PersonCRUD
from pathlib import Path

person_crud = PersonCRUD().select_person_by_id(person_id='132dfqwe2')
print(person_crud)



