from src.pipeline.face_recognition import FaceRecognition
import cv2

face_recognition = FaceRecognition()

img = cv2.imread('test/neymar.jpg')

result = face_recognition.get_recognize(img)
print(result)





