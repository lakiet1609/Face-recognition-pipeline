from src.pipeline.face_recognition import FaceRecognition
import cv2

face_recognition = FaceRecognition()

img = cv2.imread('test/ronaldo1.jpg')

result = face_recognition.get_recognize(img)
print(result)
print(type(result[0]['recognition']['score']))





