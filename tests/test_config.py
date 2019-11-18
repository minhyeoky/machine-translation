import unittest
import os
from src.config import Config

os.chdir('..')


class TestConfig(unittest.TestCase):

    def setUp(self) -> None:
        self.json_file = 'config/test_config.json'

    def test_from_to_json_file(self):
        to_json_file = 'tests/test_to_json.json'
        config = Config.from_json_file(self.json_file)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.name, 'test')
        config.to_json_file(to_json_file)
        config2 = Config.from_json_file(to_json_file)
        self.assertEqual(config.name, config2.name)

        from pprint import pprint
        pprint(config2.to_dict())



if __name__ == "__main__":
    unittest.main()
