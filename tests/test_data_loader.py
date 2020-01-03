import unittest
import os
import tensorflow as tf

from src.data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.n_data = 10
        self.validation_split = 0.5
        self.batch_size = 2
        self.data_loader = DataLoader(
            "../data/input/aihub_kor-eng/1.구어체.xlsx",
            n_data=self.n_data,
            validation_split=self.validation_split,
            deu=False,
        )
        self.data_loader_deu = DataLoader(
            "../data/input/deu.txt",
            n_data=self.n_data,
            validation_split=self.validation_split,
            deu=True,
        )

    def test_len(self):
        self.assertEqual(
            len(self.data_loader.data_train), self.n_data * (1 - self.validation_split)
        )
        self.assertEqual(
            len(self.data_loader.data_test), self.n_data * self.validation_split
        )
        self.assertEqual(
            len(self.data_loader_deu.data_train),
            self.n_data * (1 - self.validation_split),
        )

    def test_tokenizer(self):
        texts = ["<start> 나 는 매일 저녁 배트 를 만나 러 다락방 으로 가요 . <end>"]
        sequences = self.data_loader.tokenizer.tar.texts_to_sequences(
            ["<start> 나 는 매일 저녁 배트 를 만나 러 다락방 으로 가요 . <end>"]
        )
        self.assertEqual(
            self.data_loader.tokenizer.tar.sequences_to_texts(sequences), texts
        )

    def test_train_generator(self):
        it = iter(self.data_loader.train_data_generator())
        dataset = tf.data.Dataset.from_generator(
            self.data_loader.train_data_generator, output_types=(tf.int32, tf.int32)
        )
        self.assertListEqual(list(next(iter(dataset))[0].numpy()), list(next(it)[0]))
        dataset = tf.data.Dataset.from_generator(
            self.data_loader.train_data_generator, output_types=(tf.int32, tf.int32)
        ).batch(self.batch_size)
        example = next(iter(dataset))
        dataset = tf.data.Dataset.from_generator(
            self.data_loader_deu.train_data_generator, output_types=(tf.int32, tf.int32)
        ).batch(self.batch_size)
        example = next(iter(dataset))
        print(example)

    def test_test_generator(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_loader.test_data_generator, output_types=(tf.int32, tf.int32)
        ).batch(batch_size=self.batch_size)
        example = next(iter(dataset))
        print(example)
        dataset = tf.data.Dataset.from_generator(
            self.data_loader_deu.test_data_generator, output_types=(tf.int32, tf.int32)
        ).batch(batch_size=self.batch_size)
        example = next(iter(dataset))
        print(example)

    def test_dataloader_german_generator_of_train_and_test(self):
        dataset_train = tf.data.Dataset.from_generator(
            self.data_loader_deu.train_data_generator, output_types=(tf.int32, tf.int32)
        ).batch(batch_size=self.batch_size)
        example = next(iter(dataset_train))
        print(example)
        dataset_test = tf.data.Dataset.from_generator(
            self.data_loader_deu.train_data_generator, output_types=(tf.int32, tf.int32)
        ).batch(batch_size=self.batch_size)
        example = next(iter(dataset_test))
        print(example)

        print(self.data_loader_deu.tokenizer.ori.index_word)
        print(self.data_loader_deu.tokenizer.ori.num_words)
        print(self.data_loader_deu.tokenizer.ori.index_docs)
        print(self.data_loader_deu.tokenizer.ori.word_docs)


if __name__ == "__main__":
    unittest.main()
