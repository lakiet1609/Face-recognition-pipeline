from src.utils.logger import Logger
from src.database.person_database import PersonDatabase
import faiss
import numpy as np
import os


class FAISS:
    def __init__(self, config):
        self.config = config
        self.local_db = PersonDatabase.__call__()
        self.logger = Logger.__call__().get_logger()

        self.dim = self.config['dim']
        self.device = self.config['device']
        self.model_path = self.config['model_path']

        self.index = faiss.IndexFlatIP(self.dim)
        self.is_trained = False

        self.initialize()
        self.logger.info('Initialize FAISS successfully')
    
    def initialize(self):
        self.local_db.initialize_local_database()
        if not os.path.exists(self.model_path):
            return self.train()
        else:
            self.index = faiss.read_index(self.model_path)
            self.is_trained = True
            self.logger.info(f'Read model from {self.model_path}')
        
        if self.device == 'gpu':
            result = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(result, 0, self.index)
    
    def train(self):
        if (len(self.local_db.vectors.keys()) == 0):
            self.is_trained = False
            return
        
        vectors = np.array([vector for vector in self.local_db.vectors.values()], dtype=np.float32)
        self.index.add(vectors)
        self.is_trained = self.index.is_trained
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        faiss.write_index(self.index, self.model_path)
        self.logger.info(f'Write model to {self.model_path}')
    
    def search(self, embedding_vectors: np.ndarray, nearest_neighbors: int= 5):
        if not (self.is_trained or self.index.is_trained):
            return {}
        
        D, I = self.index.search(embedding_vectors, nearest_neighbors)
        people = []
        scores = []
        for i in range(len(D)):
            person_indices = []
            score_indices = []
            for j, dis in enumerate(D[i]):
                if dis > 0.5:
                    print('distance: ',dis)
                    score_indices.append(dis)
                    person_indices.append(I[i][j])
            
            people.append(person_indices)
            scores.append(score_indices)
        
        vector_ids = list(self.local_db.vectors.keys())
        recognize_results = []
        for i, people_idx in enumerate(people):
            if len(people_idx) == 0:
                recognize_results.append({'person_id': 'unrecognized'})
                continue

            person_info = []
            for j, idx in enumerate(people_idx):
                vector_id = vector_ids[idx]
                person = self.local_db.people[vector_id]
                person['score'] = float(scores[i][j])
                person_info.append(person)
            
            person_ids = [x['person_id'] for x in person_info]
            if len(set(person_ids)) == 1:
                recognize_results.append(person_info[0])
            else:
                recognize_results.append({'person_id': 'unrecognized'})
            
        return recognize_results
    
    def reload(self):
        self.local_db.initialize_local_database()
        if len(self.local_db.vectors.keys()) == 0 or len(self.local_db.people.keys()) == 0:
            self.is_trained = False
            return
        
        vectors = np.array([vector for vector in self.local_db.vectors.values()], dtype = np.float32)
        self.index.reset()
        self.index.add(vectors)
        self.is_trained = self.index.is_trained
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if self.device == "gpu":
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, self.model_path)
        else:
            faiss.write_index(self.index, self.model_path)
        
        self.logger.info(f"Rewrite model to {self.model_path}.")



