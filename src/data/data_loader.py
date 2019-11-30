from typing import NamedTuple
import pandas as pd
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .base import DataLoaderBase
from ..preprocessor.korean import KorPreprocessor
from ..preprocessor.english import EngPreprocessor

logger = tf.get_logger()
keras = tf.keras


class Data(NamedTuple):
  kor: list
  eng: list

  def __len__(self):
    l_k = len(self.kor)
    l_e = len(self.eng)
    assert l_k == l_e
    return l_k


class DataLoader(DataLoaderBase):

  def __init__(self, data_path, n_data=None, validation_split=0.1):
    super(DataLoader, self).__init__()
    logger.info('Initializing Dataloader')
    self.preprocessor = NamedTuple('Preprocessor', [('kor', KorPreprocessor),
                                                    ('eng', EngPreprocessor)])
    self.preprocessor.kor = KorPreprocessor()
    self.preprocessor.eng = EngPreprocessor()

    self.data_path = data_path
    self.n_data = n_data
    self.test_size = validation_split
    self.data_train = None
    self.data_test = None
    self.eng_vocab_size = None
    self.kor_vocab_size = None

    tokenizer = keras.preprocessing.text.Tokenizer

    self.tokenizer = NamedTuple('Tokenizer', [('kor', tokenizer),
                                              ('eng', tokenizer)])

    self.tokenizer.kor = tokenizer(num_words=None,
                                   filters='',
                                   lower=True,
                                   split=' ',
                                   oov_token='<unk>')

    self.tokenizer.eng = tokenizer(num_words=None,
                                   filters='',
                                   lower=True,
                                   split=' ',
                                   oov_token='<unk>')

    self.build()

  def build(self):
    self._load_data()

  def train_data_generator(self):
    en, ko = self.data_train.eng, self.data_train.kor
    for e, k in zip(en, ko):
      yield e, k

  def test_data_generator(self):
    en, ko = self.data_test.eng, self.data_test.kor
    for e, k in zip(en, ko):
      yield e, k

  @staticmethod
  def _tokenize(texts, tokenizer, fit):
    if fit:
      tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, padding='post', truncating='post',
        value=0)    # padding value
    return sequences

  def _load_data(self):
    logger.info(f'Loading data from {self.data_path}')
    data = pd.read_excel(self.data_path, sheet_name='Sheet1')
    logger.info(data.head())

    _data = []
    for idx, row in data.iterrows():
      eng = row['en']
      kor = row['ko']

      eng = self.preprocessor.eng.preprocess(eng)
      kor = self.preprocessor.kor.preprocess(kor)

      _data.append([eng, kor])

      if self.n_data:
        if self.n_data == len(_data):
          break

    _data_train, _data_test = train_test_split(_data,
                                               test_size=self.test_size,
                                               shuffle=False)
    en_train, ko_train = zip(*_data_train)
    en_test, ko_test = zip(*_data_test)

    en_train = self._tokenize(en_train, self.tokenizer.eng, fit=True)
    ko_train = self._tokenize(ko_train, self.tokenizer.kor, fit=True)

    en_test = self._tokenize(en_test, self.tokenizer.eng, fit=False)
    ko_test = self._tokenize(ko_test, self.tokenizer.kor, fit=False)

    self.eng_vocab_size = len(self.tokenizer.eng.word_index) + 1
    self.kor_vocab_size = len(self.tokenizer.kor.word_index) + 1

    logger.info(f' English vocab size: {self.eng_vocab_size}')
    logger.info(f' Korean vocab size: {self.kor_vocab_size}')

    self.data_train = Data(kor=ko_train, eng=en_train)
    self.data_test = Data(kor=ko_test, eng=en_test)
