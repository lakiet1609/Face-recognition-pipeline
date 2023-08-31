import yaml
import json

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def read_yaml(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def write_json(file):
    with open('database.json', 'w') as f:
        json.dump(file, f)


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data
