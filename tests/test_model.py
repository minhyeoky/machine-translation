import os
import unittest
import tensorflow as tf

from src.model.model import Encoder, Decoder


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = tf.constant([
            [9, 2, 3, 1, 0],
            [1, 2, 1, 3, 4]
        ])
        self.batch_size = 2
        self.vocab_size = 10
        self.embedding_size = 128
        self.n_units = 128

    def test_decoder(self):
        encoder = Encoder(vocab_size=self.vocab_size, embedding_size=self.embedding_size, n_units=self.n_units)
        inputs = self.test_data, encoder.initial_state(self.batch_size)
        outputs, h, c = encoder(inputs)
        decoder = Decoder(vocab_size=self.vocab_size, embedding_size=self.embedding_size, n_units=self.n_units)
        initial_state = h, c
        for t in range(1, len(outputs)):
            output, h, c = decoder((tf.expand_dims(self.test_data[:, t], 1), initial_state))
            initial_state = h, c

            self.assertEqual(output.shape, (self.batch_size, self.vocab_size))
            self.assertEqual(h.shape, (self.batch_size, self.n_units))
            self.assertEqual(c.shape, (self.batch_size, self.n_units))

    def test_encoder(self):
        encoder = Encoder(vocab_size=10, embedding_size=128, n_units=128)
        inputs = self.test_data, encoder.initial_state(2)
        outputs, h, c = encoder(inputs, training=True)
        self.assertEqual(outputs.shape, (self.batch_size, 5, self.n_units))
        self.assertEqual(h.shape, (self.batch_size, self.n_units))
        self.assertEqual(c.shape, (self.batch_size, self.n_units))


if __name__ == '__main__':
    unittest.main()
