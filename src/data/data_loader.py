from typing import NamedTuple
import pandas as pd
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .base import DataLoaderBase
from ..preprocessor.korean import KorPreprocessor
from ..preprocessor.english import EngPreprocessor

logger = logging.getLogger(__name__)
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
    def __init__(self, data_path, n_data=None, test_size=0.1):
        super(DataLoader, self).__init__()
        logger.info('Initiating Dataloader')
        self.preprocessor = NamedTuple('Preprocessor', [('kor', KorPreprocessor),
                                                        ('eng', EngPreprocessor)])
        self.preprocessor.kor = KorPreprocessor()
        self.preprocessor.eng = EngPreprocessor()

        self.data_path = data_path
        self.n_data = n_data
        self.test_size = test_size
        self.data_train = None
        self.data_test = None

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
        self._load_data()

    def train_data_generator(self):
        en, ko = self.data_train.eng, self.data_train.kor
        for e, k in zip(en, ko):
            yield e, k

    def test_data_generator(self):
        en, ko = self.data_test.eng, self.data_test.eng
        for e, k in zip(en, ko):
            yield e, k

    @staticmethod
    def _tokenize(texts, tokenizer):
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                                  padding='post')
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

        en_train = self._tokenize(en_train, self.tokenizer.eng)
        ko_train = self._tokenize(ko_train, self.tokenizer.kor)

        en_test = self.tokenizer.eng.texts_to_sequences(en_test)
        ko_test = self.tokenizer.kor.texts_to_sequences(ko_test)

        self.data_train = Data(kor=ko_train, eng=en_train)
        self.data_test = Data(kor=ko_test, eng=en_test)
