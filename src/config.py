import json
import logging

logger = logging.getLogger(__name__)


class Config:

  def __init__(self, **kwargs):
    logger.info('Loading Config')
    # Name
    self.name = kwargs.get('name', None)

    # Meta
    self.meta: dict = kwargs['meta']
    self.display_step = self.meta.get('display_step', None)
    self.save_step = self.meta.get('save_step', None)
    self.ckpt_dir = self.meta.get('ckpt_dir', None)
    self.ckpt_max_keep = self.meta.get('ckpt_max_keep', None)
    self.logdir = self.meta.get('logdir', None)
    self.log_level = self.meta.get('log_level', None)

    # Train
    self.train: dict = kwargs['train']
    self.batch_size = self.train.get('batch_size', None)
    self.epochs = self.train.get('epochs', None)
    # Train - data_loader
    self.data_loader: dict = kwargs['data_loader']
    self.batch_size = self.data_loader.get('batch_size', None)
    self.buffer_size = self.data_loader.get('buffer_size', self.batch_size * 10)

    # Train - optimizer
    self.optimizer = self.train.get('optimizer', None)

    # Train - model
    self.model = self.train.get('model', None)
    self.encoder = self.model.get('encoder', None)
    self.decoder = self.model.get('decoder', None)

    # Inference
    self.inference: dict = kwargs['inference']
    self.inference_size = self.inference.get('inference_size', None)

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
