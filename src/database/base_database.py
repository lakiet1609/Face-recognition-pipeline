import pymongo
from datetime import datetime
from src.utils.logger import Logger

class BaseDatabase(object):
    def __init__(self, config):
        self.hostname = config['hostname']