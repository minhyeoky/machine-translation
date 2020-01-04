from .data_loader import DataLoader as DataLoaderBase
from .data_loader import Data
from typing import NamedTuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ..preprocessor.base import PreprocessorBase
from ..preprocessor.english import EngPreprocessor
from ..preprocessor.german import GerPreprocessor
from ..preprocessor.korean import KorPreprocessor

OOV_TOKEN = "<UNK>"
logger = tf.get_logger()

keras = tf.keras


class DataLoader(DataLoaderBase):
    def __init__(self, *args, **kwargs):
        # super(DataLoader, self).__init__(*args, **kwargs)

        num_words = kwargs.get("num_words", None)
        maxlen = kwargs.get("maxlen", None)
        deu = kwargs.get("deu", None)
        n_data = kwargs.get("n_data", None)
        validation_split = kwargs.get("validation_split", None)

        self.maxlen = maxlen
        self.deu = deu
        self.num_words = num_words
        logger.info("Initializing Dataloader")
        self.preprocessor = NamedTuple(
            "Preprocessor", [("ori", PreprocessorBase), ("tar", PreprocessorBase)]
        )
        self.preprocessor.ori = EngPreprocessor()
        if deu:
            self.preprocessor.tar = GerPreprocessor()
        else:
            self.preprocessor.tar = KorPreprocessor()

        self.data_path = args[0]
        self.n_data = n_data
        self.test_size = validation_split
        self.data_train = None
        self.data_test = None
        self.ori_vocab_size = None
        self.tar_vocab_size = None

        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=num_words, filters="", lower=True, split=" ", oov_token=OOV_TOKEN
        )

        self.build()

    def _load_data(self):
        logger.info(f"Loading data from {self.data_path}")
        _data = []
        if self.deu:
            with open(self.data_path, "r", encoding="utf8") as f:
                for line in f:
                    line = line.split("\t")
                    ori = self.preprocessor.ori.preprocess(line[0])
                    tar = self.preprocessor.tar.preprocess(line[1])
                    _data.append([ori, tar])
                    if self.n_data:
                        if self.n_data == len(_data):
                            break
        else:
            data = pd.read_excel(self.data_path, sheet_name="Sheet1")
            for idx, row in data.iterrows():
                en = row["en"]
                ko = row["ko"]

                en = self.preprocessor.ori.preprocess(en)
                ko = self.preprocessor.tar.preprocess(ko)

                _data.append([en, ko])

                if self.n_data:
                    if self.n_data == len(_data):
                        break

        _data_train, _data_test = train_test_split(
            _data, test_size=self.test_size, shuffle=False
        )
        ori_train, tar_train = zip(*_data_train)
        ori_test, tar_test = zip(*_data_test)

        # Main difference.
        self.tokenizer.fit_on_texts(ori_train + tar_train)

        ori_train = self._tokenize(ori_train, self.tokenizer, fit=False)
        tar_train = self._tokenize(tar_train, self.tokenizer, fit=False)

        ori_test = self._tokenize(ori_test, self.tokenizer, fit=False)
        tar_test = self._tokenize(tar_test, self.tokenizer, fit=False)

        self.data_train = Data(ori=ori_train, tar=tar_train)
        self.data_test = Data(ori=ori_test, tar=tar_test)
