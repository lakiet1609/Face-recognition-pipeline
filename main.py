from src.pipeline.face_recognition import FaceRecognition
from src.manger.detection_manager import DetectionManager
from src.config.configuration import Configuration
from src.utils.common import read_yaml
import cv2

img = cv2.imread('faces/CR7/bd10d9ec07894146bc8884bc161de5c8.jpg')

face_recognition = FaceRecognition()
ret = face_recognition.get_recognize(img)
print(ret)
