import unittest
from src.config import Config
from src.utils import get_sentence, get_bleu_score


class TestConfig(unittest.TestCase):

    def setUp(self) -> None:
        self.json_file = '../config/test_config.json'

    def test_from_to_json_file(self):
        to_json_file = '../tests/test_to_json.json'
        config = Config.from_json_file(self.json_file)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.name, 'test')
        config.to_json_file(to_json_file)
        config2 = Config.from_json_file(to_json_file)
        self.assertEqual(config.name, config2.name)

        from pprint import pprint
        pprint(config2.to_dict())

    def test_utils_get_sentence(self):
        orig = ['<start>', '안녕', '나는', '이민혁', '.', '<end>', '.']
        sentence = '안녕 나는 이민혁 .'
        self.assertEqual(get_sentence(x=orig), sentence)


if __name__ == "__main__":
    unittest.main()
