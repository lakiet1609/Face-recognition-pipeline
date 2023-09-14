from src.utils.logger import Logger
from src.components.scrfd import SCRFD
from src.components.tensorrt.scrfd_trt import SCRFD_TRT

class DetectionManager:
    def __init__(self, config, engine):
        self.engine = engine
        self.config = config
        self.logger = Logger.__call__().get_logger()
        
        if self.engine == 'scrfd_onnx':
            self.engine = SCRFD(config)
        elif self.engine == 'scrfd_trt':
            self.engine = SCRFD_TRT(config)
        else:
            self.logger.error(f'Does not support {self.engine} engine')
            exit(0)

    def get_engine(self):
        return self.engine
    

        