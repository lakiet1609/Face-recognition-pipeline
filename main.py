from src.components.arcface import ARCFACE
from src.components.scrfd import SCRFD
from src.config.configuration import Configuration
from src.utils.face_det import Face
import cv2


scrfd_config = Configuration().get_face_detection()
scrfd_process = SCRFD(scrfd_config)

arcface_config = Configuration().get_face_embedding()
arcface_process = ARCFACE(arcface_config)

image = cv2.imread('test/neymar.jpg')

predicted= scrfd_process.detect(img=image, input_size=[640, 640])

bounding_box = predicted[0].tolist()[0]
keypoints = predicted[1][0]

face = Face(bbox= bounding_box[:4], kps= keypoints, det_score = bounding_box[4])
vector = arcface_process.get(image, face)
print(f'Face embeddings shape: {vector.shape}')


