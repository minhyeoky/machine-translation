import re
from .base import PreprocessorBase


class EngPreprocessor(PreprocessorBase):
    def __init__(self):
        super(EngPreprocessor, self).__init__()
        pass

    @staticmethod
    def _clean(s):
        s = re.sub(r"[^a-zA-Z?.!,]", " ", s)
        s = s.strip()
        return s

    def preprocess(self, s: str):
        s = self._basic_nmt(s)
        s = self._clean(s)
        s = self._add_token(s)
        return s
