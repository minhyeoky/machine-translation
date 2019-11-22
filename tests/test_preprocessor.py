import os
import unittest
from typing import NamedTuple

from src.preprocessor.english import EngPreprocessor
from src.preprocessor.korean import KorPreprocessor


class TestPreprocessor(unittest.TestCase):

    def setUp(self) -> None:
        self.preprocessor = NamedTuple('Preprocessor', [('kor', KorPreprocessor), ('eng', EngPreprocessor)])
        self.preprocessor.kor = KorPreprocessor()
        self.preprocessor.eng = EngPreprocessor()

    def test_eng_preprocessor(self):
        self.assertEqual(self.preprocessor.eng.preprocess('he is a boy.'), '<start> he is a boy . <end>')
        self.assertEqual(self.preprocessor.eng.preprocess('May I borrow this book?'),
                         '<start> may i borrow this book ? <end>')

    def test_kor_preprocessor(self):
        self.assertEqual(self.preprocessor.kor.preprocess('안녕하세요?'), '<start> 안녕 하 세요 ? <end>')
        self.assertEqual(self.preprocessor.kor.preprocess('제가 이 책을 빌려도 되나요?'), '<start> 제 가 이 책 을 빌려 도 되 나요 ? <end>')


if __name__ == "__main__":
    unittest.main()
