from src.database.base_database import BaseDatabase
from src.utils.common import Singleton
from src.config.configuration import Configuration
import numpy as np

class PersonDatabase(BaseDatabase, metaclass=Singleton):
    def __init__(self):
        self.config = Configuration.__call__().get_database()
        super(PersonDatabase, self).__init__(self.config)

        database_name = self.config['database_name']
        collection_name = self.config['collection_name']
        self.database = self.client[database_name]
        self.people_collection = self.database[collection_name]
    
    def get_people_collection(self):
        return self.people_collection
    
    def initialize_local_database(self):
        people_docs = self.people_collection.find()
        self.vectors = {}
        self.people = {}
        for person in people_docs:
            person_id = person['id']
            person_name = person['name']
            if ('faces' not in person.keys()) or (person['faces'] is None):
                continue
            for face in person['faces']:
                if ('vector' not in face.keys()) or (face['vector'] is None):
                    continue
                vector_id = face['id']
                vector = np.array(face['vector'], dtype = np.float32)
                self.vectors[vector_id] = vector / np.linalg.norm(vector)
                self.people[vector_id] = {
                    'person_id': person_id,
                    'person_name': person_name
                }

    def get_local_database(self):
        return self.people, self.vectors
    
    def check_person_by_id(self, person_id: str) -> bool:
        person_doc = self.people_collection.find_one({'id': person_id}, {'_id': 0})
        if person_doc is None:
            return False
        return True
    
    def check_face_by_id(self, person_id: str, face_id: str) -> bool:
        people_doc = self.people_collection.find_one({'id': person_id}, {'_id': 0, 'faces.vector': 0})
        if people_doc is None:
            return False
        if 'faces' not in people_doc.keys():
            return False
        elif people_doc['faces'] is None:
            return False
        elif len(people_doc['faces']) == 0:
            return False
        current_id = [x['id'] for x in people_doc['faces']]
        if face_id not in current_id:
            return False
        return True

    
