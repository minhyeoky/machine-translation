"""
Abstract base class for DataLoader
"""
from abc import *


class DataLoaderBase(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def data_generator(self):
        pass
