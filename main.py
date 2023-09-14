from src.components.tensorrt.scrfd_trt import SCRFD_TRT
from src.components.scrfd import SCRFD
from src.utils.common import read_yaml
import cv2

img = cv2.imread('faces/CR7/bd10d9ec07894146bc8884bc161de5c8.jpg')

config = read_yaml('configs/scrfd_trt_cfg.yaml')
scrfd_trt = SCRFD_TRT(config)
scrfd_trt.detect(img)




