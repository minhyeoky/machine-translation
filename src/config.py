import json
import logging

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, **kwargs):
        logger.info('Loading Config')

        # Meta
        self.name = kwargs.get('name', None)
        self.display_step = kwargs.get('display_step', 10)

        # Hyperparamters
        self.batch_size = kwargs.get('batch_size', None)
        self.epochs = kwargs.get('epochs', None)
        self.lr = kwargs.get('lr', None)
        self.buffer_size = kwargs.get('buffer_size', self.batch_size * 100)
        self.beta_1 = kwargs.get('beta_1', 0.9)
        self.embedding_size = kwargs.get('embedding_size', 256)
        self.n_units = kwargs.get('n_units', 256)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, mode='r', encoding='utf8') as f:
            text = f.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path):
        with open(json_file_path, mode='w', encoding='utf8') as f:
            f.write(self.to_json())
