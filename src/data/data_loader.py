from typing import NamedTuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .base import DataLoaderBase
from ..preprocessor.english import EngPreprocessor
from ..preprocessor.german import GerPreprocessor
from ..preprocessor.korean import KorPreprocessor

OOV_TOKEN = '<unk>'

logger = tf.get_logger()
keras = tf.keras


class Data(NamedTuple):
  ori: list
  tar: list

  def __len__(self):
    l_o = len(self.ori)
    l_t = len(self.tar)
    assert l_t == l_o
    return l_t


class DataLoader(DataLoaderBase):

  def __init__(self,
               data_path,
               n_data=None,
               validation_split=0.1,
               deu=False,
               num_words=None,
               maxlen=None):
    super(DataLoader, self).__init__()
    self.maxlen = maxlen
    self.deu = deu
    logger.info('Initializing Dataloader')
    self.preprocessor = NamedTuple('Preprocessor', [('ori', EngPreprocessor),
                                                    ('tar', GerPreprocessor)])
    self.preprocessor.ori = EngPreprocessor()
    if deu:
      self.preprocessor.tar = GerPreprocessor()
    else:
      self.preprocessor.tar = KorPreprocessor()

    self.data_path = data_path
    self.n_data = n_data
    self.test_size = validation_split
    self.data_train = None
    self.data_test = None
    self.ori_vocab_size = None
    self.tar_vocab_size = None

    tokenizer = keras.preprocessing.text.Tokenizer

    self.tokenizer = NamedTuple('Tokenizer', [('ori', tokenizer),
                                              ('tar', tokenizer)])

    self.tokenizer.ori = tokenizer(num_words=num_words,
                                   filters='',
                                   lower=True,
                                   split=' ',
                                   oov_token=OOV_TOKEN)

    self.tokenizer.tar = tokenizer(num_words=num_words,
                                   filters='',
                                   lower=True,
                                   split=' ',
                                   oov_token=OOV_TOKEN)

    self.build()

  def build(self):
    self._load_data()

  def train_data_generator(self):
    ori, tar = self.data_train.ori, self.data_train.tar
    for o, t in zip(ori, tar):
      yield o, t

  def test_data_generator(self):
    ori, tar = self.data_test.ori, self.data_test.tar
    for o, t in zip(ori, tar):
      yield o, t

  def _tokenize(self, texts, tokenizer, fit):
    if fit:
      tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=self.maxlen,
        padding='post',
        truncating='post',
        value=0)    # padding value
    return sequences

  def _load_data(self):
    logger.info(f'Loading data from {self.data_path}')
    _data = []
    if self.deu:
      with open(self.data_path, 'r', encoding='utf8') as f:
        for line in f:
          line = line.split('\t')
          ori = self.preprocessor.ori.preprocess(line[0])
          tar = self.preprocessor.tar.preprocess(line[1])
          _data.append([ori, tar])
          if self.n_data:
            if self.n_data == len(_data):
              break
    else:
      data = pd.read_excel(self.data_path, sheet_name='Sheet1')
      for idx, row in data.iterrows():
        en = row['en']
        ko = row['ko']

        en = self.preprocessor.ori.preprocess(en)
        ko = self.preprocessor.tar.preprocess(ko)

        _data.append([en, ko])

        if self.n_data:
          if self.n_data == len(_data):
            break

    _data_train, _data_test = train_test_split(_data,
                                               test_size=self.test_size,
                                               shuffle=False)
    ori_train, tar_train = zip(*_data_train)
    ori_test, tar_test = zip(*_data_test)

    ori_train = self._tokenize(ori_train, self.tokenizer.ori, fit=True)
    tar_train = self._tokenize(tar_train, self.tokenizer.tar, fit=True)

    ori_test = self._tokenize(ori_test, self.tokenizer.ori, fit=False)
    tar_test = self._tokenize(tar_test, self.tokenizer.tar, fit=False)

    self.ori_vocab_size = len(self.tokenizer.ori.word_index) + 1
    self.tar_vocab_size = len(self.tokenizer.tar.word_index) + 1

    logger.info(f' Original vocab size: {self.ori_vocab_size}')
    logger.info(f' Target vocab size: {self.tar_vocab_size}')

    self.data_train = Data(ori=ori_train, tar=tar_train)
    self.data_test = Data(ori=ori_test, tar=tar_test)
