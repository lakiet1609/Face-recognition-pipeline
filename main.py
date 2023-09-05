from src.pipeline.face_recognition import FaceRecognition
import cv2

img = cv2.imread('test/ronaldo1.jpg') 

detection = FaceRecognition().get_face_encode(img=img, largest_box=True)
print(detection)