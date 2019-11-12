"""
Abstract base class for Preprocessor
"""
import re
import unicodedata
from abc import *


class PreprocessorBase(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, s):
        """

        :param s:
        :return:
        """
        pass

    @staticmethod
    def _add_token(s):
        s = '<start> ' + s + ' <end>'
        return s

    @staticmethod
    def _basic_nmt(s):
        s = s.lower()
        s = s.strip()

        s = re.sub(r'([?.!,])', r' \1', s)
        s = re.sub(r'[" "]', ' ', s)
        return s

    @staticmethod
    def _normalize(s, form='NFC'):
        """
        Unicode normalization
        :param s: sentence
        :param form:
            https://www.wikiwand.com/ko/%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C_%EC%A0%95%EA%B7%9C%ED%99%94
        """
        return unicodedata.normalize(form, s)

    @staticmethod
    def _is_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((33 <= cp <= 47) or (58 <= cp <= 64) or
                (91 <= cp <= 96) or (123 <= cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    @staticmethod
    def _is_control(char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    @staticmethod
    def _is_whitespace(char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    @staticmethod
    def _is_chinese_char(char):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        cp = ord(char)
        if ((0x4E00 <= cp <= 0x9FFF) or
                (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or
                (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or
                (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False

    @staticmethod
    def _is_korean_char(char):
        """Checks whether CP is the codepoint of a KOR character."""
        # This defines a "Hangul Jamo" as anything in
        # Hangul Jamo & Hangul Compatibility Jamo Unicode block:
        # https://en.wikipedia.org/wiki/Hangul_Jamo_(Unicode_block)
        # https://en.wikipedia.org/wiki/Hangul_Compatibility_Jamo
        cp = ord(char)
        if ((0x1100 <= cp <= 0x11FF) or
                (0x3131 <= cp <= 0x319E)):
            return True

        return False
