import os
import unittest
import tensorflow as tf
import numpy as np

from src.model.seq2seq import Encoder, Decoder
from src.model.seq2seq_bahdanau import BahdanauAttention
from src.model.seq2seq_bahdanau import Encoder as BahdanauEncoder
from src.model.seq2seq_bahdanau import Decoder as BahdanauDecoder
from src.model.seq2seq_bidirectional import Encoder as Seq2seqBidirectionalEncoder


class MyTestCase(unittest.TestCase):

  def setUp(self) -> None:
    self.test_data = np.random.randint(0, 9, size=(2, 5))

    self.batch_size = 2
    self.max_seq = 5
    self.vocab_size = 10
    self.embedding_size = 128
    self.n_units = 128

  def test_decoder(self):
    encoder = Encoder(vocab_size=self.vocab_size,
                      embedding_size=self.embedding_size,
                      n_units=self.n_units)
    inputs = self.test_data, encoder.initial_state(self.batch_size)
    outputs, h, c = encoder(inputs)
    decoder = Decoder(vocab_size=self.vocab_size,
                      embedding_size=self.embedding_size,
                      n_units=self.n_units)
    initial_state = h, c
    for t in range(1, len(outputs)):
      output, h, c = decoder((tf.expand_dims(self.test_data[:, t],
                                             1), initial_state))
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

  def test_encoder_bahdanau(self):
    encoder = BahdanauEncoder(self.vocab_size, self.embedding_size,
                              self.n_units)
    inputs = self.test_data, encoder.initial_state(self.batch_size)
    outputs, h, c = encoder(inputs, training=True)

    self.assertEqual(outputs.shape, (self.batch_size, 5, self.n_units))
    self.assertEqual(h.shape, (self.batch_size, self.n_units))
    self.assertEqual(c.shape, (self.batch_size, self.n_units))

  def test_decoder_bahdanau(self):
    encoder = BahdanauEncoder(self.vocab_size, self.embedding_size,
                              self.n_units)
    inputs = self.test_data, encoder.initial_state(self.batch_size)
    outputs, h, c = encoder(inputs, training=True)

  def test_encoder_seq2seq_bidirectional(self):
    encoder = Seq2seqBidirectionalEncoder(self.vocab_size, self.embedding_size,
                                          self.n_units)
    inputs = self.test_data, encoder.initial_state(self.batch_size)
    outputs, h_f, h_b, c_f, c_b = encoder(inputs, training=True)

    self.assertEqual(outputs.shape,
                     (self.batch_size, self.max_seq, self.n_units * 2))
    self.assertEqual(h_f.shape, (self.batch_size, self.n_units))
    self.assertEqual(h_f.shape, h_b.shape)
    self.assertEqual(c_f.shape, h_f.shape)





if __name__ == '__main__':
  unittest.main()
