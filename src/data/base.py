"""
Abstract base class for DataLoader
"""
from abc import *


class DataLoaderBase(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train_data_generator(self):
        pass

    @abstractmethod
    def test_data_generator(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass


class TokenizerBase(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self):
        pass
