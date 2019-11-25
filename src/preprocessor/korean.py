from konlpy.tag import Mecab
from collections import namedtuple
import re
from .base import PreprocessorBase


class KorPreprocessor(PreprocessorBase):

  def __init__(self):
    super(KorPreprocessor, self).__init__()

    self.tagger = Mecab()

  def _to_morphs(self, s):
    return self.tagger.pos(s)

  @staticmethod
  def _clean(s):
    s = re.sub(r'[^가-힣ㄱ-ㅎ?.!,]', ' ', s)
    s = s.strip()
    return s

  def preprocess(self, s):
    s = self._basic_nmt(s)
    s = self._clean(s)

    tagged = self._to_morphs(s)

    _s = []
    for w, _ in tagged:
      _s.append(w)
    s = ' '.join(_s)
    s = self._add_token(s)
    return s
