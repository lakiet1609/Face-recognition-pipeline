from src.utils.logger import Logger
from src.utils.face_align import norm_crop, estimate_norm
import onnxruntime as ort
import cv2

class ARCFACE:
    def __init__(self, config):
        self.config = config
        self.model_path = self.config['model_path']
        self.device = self.config['device']
        self.input_names = self.config['input_names']
        self.input_size = self.config['input_size']
        self.output_names = self.config['output_names']
        self.output_shape = self.config['output_shape']
        self.threshold = self.config['threshold']
        self.input_mean = self.config['input_mean']
        self.input_std = self.config['input_std']
        self.logger = Logger.__call__().get_logger()

        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif self.device == 'cuda':
            providers = ['CUDAExecutionProvider']
        else:
            self.logger.info(f'Does not support this {self.device}')
            exit(0)
        
        self.sess = ort.InferenceSession(self.model_path, providers = providers)
        self.logger.info('Initialize ARCFACE Onnx successfully')


    def get(self, img, face):
        align_img = norm_crop(img=img, landmark=face.kps)
        face.embedding = self.get_features(align_img).flatten()
        return face.normed_embedding

    def get_features(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        blob = cv2.dnn.blobFromImages(imgs, 1.0/self.input_std, self.input_size, 
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB = True)
        
        net_out = self.sess.run(self.output_names, {self.input_names[0]: blob})[0]

        return net_out