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
    
    def search(self, embedding_vectors: np.ndarray, nearest_neighgbors: int = 1):
        # check if model is trained completely before searching
        if not (self.is_trained or self.index.is_trained):
            return {}
        # search nearest embedding vectors
        D, I = self.index.search(embedding_vectors, nearest_neighgbors) 
        people = []
        scores = []
        # select embedding vectors with cosin > threshold
        for i in range (len(D)):
            person_indexes = []
            scores_indexes = []
            for j, dis in enumerate(D[i]):
                if dis > 0.5:
                    scores_indexes.append(float(dis))
                    person_indexes.append(I[i][j])
            people.append(person_indexes)
            scores.append(scores_indexes)
        
        vector_ids = list(self.local_db.vectors.keys())
        recognize_results = []
        
        for i, person_indexes in enumerate(people):
            if len(person_indexes) == 0: 
                recognize_results.append({"person_id": "unrecognize"})   
                continue
            person_infos = []
            for j,index in enumerate(person_indexes):
                print()
                vector_id = vector_ids[index]
                person = self.local_db.people[vector_id]
                person ["score"] = scores[i][j]
                person_infos.append(person)
            # check if all results are only one person's
            person_ids = [x["person_id"] for x in person_infos]
            if len(set(person_ids)) == 1:
                recognize_results.append(person_infos[0])
            else:
                recognize_results.append({"person_id": "unrecognize"}) 

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



