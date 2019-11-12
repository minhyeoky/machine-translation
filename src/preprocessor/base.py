"""
Abstract base class for Preprocessor
"""

from abc import *


class PreprocessorBase(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass
