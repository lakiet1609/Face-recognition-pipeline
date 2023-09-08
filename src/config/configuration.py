from src.utils.common import Singleton, read_yaml

class Configuration(object, metaclass=Singleton):
    def __init__(self):
        self.config = 'configs/face_recognition.yaml'
        self.face_detection = None
        self.face_embedding = None
        self.faiss = None
        self.database = None
        self.init_config()
    
    def init_config(self):
        config = read_yaml(self.config)
        self.face_detection = read_yaml(config['face_detection'])
        self.face_embedding = read_yaml(config['face_embedding'])
        self.faiss = read_yaml(config['faiss'])
        self.database = read_yaml(config['database'])
    
    def get_face_detection(self):
        return self.face_detection
    
    def get_face_embedding(self):
        return self.face_embedding
    
    def get_database(self):
        return self.database
    
    def get_faiss(self):
        return self.faiss