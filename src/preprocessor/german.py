import re

from .base import PreprocessorBase


class GerPreprocessor(PreprocessorBase):

  def __init__(self):
    super(GerPreprocessor, self).__init__()

  @staticmethod
  def _clean(s):
    s = re.sub(r'[^a-zA-Z?.!,]', ' ', s)
    s = s.strip()
    return s

  def preprocess(self, s: str):
    s = self.unicode_to_ascii(s)
    s = self._basic_nmt(s)
    s = self._clean(s)
    s = self._add_token(s)
    return s
