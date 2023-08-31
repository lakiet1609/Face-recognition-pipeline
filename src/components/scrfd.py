from src.utils.logger import Logger
from src.utils.common import read_yaml
from src.utils.face_det import nms, distance2bbox, distance2kps, custom_resize
import numpy as np
import cv2
import onnxruntime as ort

class SCRFD:
    def __init__(self, config):
        self.config = config
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']
        self.model_path = self.config['model_path']


