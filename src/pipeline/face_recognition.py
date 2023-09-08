from src.utils.logger import Logger
from src.utils.common import Singleton
from src.config.configuration import Configuration
from src.components.arcface import ARCFACE
from src.components.scrfd import SCRFD
from src.components.faiss import FAISS
from src.utils.face_det import Face
import numpy as np

class FaceRecognition(metaclass=Singleton):
    def __init__(self):
        self.config = Configuration.__call__()
        self.face_detection_config = self.config.get_face_detection()
        self.face_embedding_config = self.config.get_face_embedding()
        self.faiss_config = self.config.get_faiss()
        
        self.face_detection = SCRFD(self.face_detection_config)
        self.face_embedding = ARCFACE(self.face_embedding_config)
        self.faiss = FAISS(self.faiss_config)

        self.logger = Logger.__call__().get_logger()
    
    def get_face_detect(self):
        return self.face_detection
    
    def get_face_embedding(self):
        return self.face_embedding
    
    def get_face_encode(self, img: np.array, largest_box: False) -> np.array:
        detection = self.face_detection.detect(img)
        
        self.logger.info('Detected face by SCRFD completed !')
        
        if detection[0].shape[0] == 0 or detection[1].shape[0] == 0:
            self.logger.info('No face detected by SCRFD !')
            return np.array([])
        
        if largest_box:
            detection = self.get_largest_box(detection)
        
        vectors_list = []
        for i, det in enumerate(detection[0]):
            face = Face(bbox= det[:4], kps= detection[1][i], det_score = det[4])
            vector = self.face_embedding.get(img, face)
            vectors_list.append(vector)
        
        self.logger.info('Get face keypoints by ARCFACE completed !')
        
        results = [detection[0], detection[1], vectors_list]
        return results

    def get_largest_box(self, detection):
        bboxes = detection[0]
        kps = detection[1]
        largest_box = max(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        kps_corr = kps[np.where((bboxes==largest_box).all(axis=1))]
        detection_largest = (np.expand_dims(largest_box, axis=0), kps_corr)
        return detection_largest
    
    def get_recognize(self, image: np.array):
        dets, kps, encodes = self.get_face_encode(image, True)
        recognize = self.faiss.search(np.array(encodes))
        results = self.convert_result_to_dict(dets, kps, recognize)
        return results
    
    def convert_result_to_dict(self, det, kps, recognize):
        recognition_results = []
        for i in range(len(det)):
            face = {}
            bbox = list(det[i][:4])
            det_score = det[i][4]
            landmarks = []
            for point in kps[i]:
                landmarks.append(list(point))
            
            face['bbox'] = bbox
            face['score'] = det_score
            face['landmarks'] = landmarks
            face['recognition'] = recognize[i]
            recognition_results.append(face)
        return recognition_results

    
