import unittest
import os
import tensorflow as tf

from src.data.data_loader import DataLoader

os.chdir('..')


class TestDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.data_loader = DataLoader('./data/input/aihub_kor-eng/1.구어체.xlsx', n_data=5, test_size=0.2)

    def test_len(self):
        self.assertEqual(len(self.data_loader.data_train), 4)
        self.assertEqual(len(self.data_loader.data_test), 1)

    def test_tokenizer(self):
        texts = ['<start> 나 는 매일 저녁 배트 를 만나 러 다락방 으로 가요 . <end>']
        sequences = self.data_loader.tokenizer.kor.texts_to_sequences(['<start> 나 는 매일 저녁 배트 를 만나 러 다락방 으로 가요 . <end>'])
        self.assertEqual(self.data_loader.tokenizer.kor.sequences_to_texts(sequences), texts)

    def test_generator(self):
        it = iter(self.data_loader.train_data_generator())
        dataset = tf.data.Dataset.from_generator(self.data_loader.train_data_generator, output_types=(tf.int32, tf.int32))
        self.assertListEqual(list(next(iter(dataset))[0].numpy()), list(next(it)[0]))


if __name__ == "__main__":
    unittest.main()
