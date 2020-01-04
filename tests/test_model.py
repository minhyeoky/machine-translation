import os
import unittest
import tensorflow as tf
import numpy as np

from src.model.seq2seq import Encoder, Decoder
from src.model.seq2seq_bahdanau import BahdanauAttention
from src.model.seq2seq_bahdanau import Encoder as BahdanauEncoder
from src.model.seq2seq_bahdanau import Decoder as BahdanauDecoder
from src.model.seq2seq_bidirectional import Encoder as Seq2seqBidirectionalEncoder
from src.model.transformer import ScaledDotProductAttention, MultiHeadAttention
from src.utils import create_pad_mask
from src.model.transformer import Encoder as TransformerEncoder
from src.model.transformer import DecoderLayer as TransformerDecoderLayer
from src.model.transformer import Decoder as TransformerDecoder
from src.data.data_loader import DataLoader
from src.model.transformer import PositionalEmbedding


class MyTestCase(unittest.TestCase):

  def setUp(self) -> None:
    self.test_data = np.random.randint(0, 9, size=(2, 5))

    self.batch_size = 2
    self.max_seq = 5
    self.vocab_size = 10
    self.embedding_size = 128
    self.n_units = 128
    # yapf: disable
    self.x_for_mask = tf.constant(
        [[2, 3, 4, 5, 0],
         [1, 2, 3, 4, 0],
         [4, 2, 0, 0, 0]], dtype=tf.int32)

    ###
    # https://www.tensorflow.org/tutorials/text/transformer#positional_encoding
    self.q = tf.constant([[[0, 10, 0]]], dtype=tf.float32)  # (1, 3)
    self.k = tf.constant([[[10, 0, 0],
                           [0, 10, 0],
                           [0, 0, 10],
                           [0, 0, 10]]],
                         dtype=tf.float32)  # (4, 3)
    self.v = tf.constant([[[1, 0],
                           [10, 0],
                           [100, 5],
                           [1000, 6]]], dtype=tf.float32)  # (4, 3)
    # yapf: enable

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

  # def test_encoder(self):
  #   encoder = Encoder(vocab_size=10, n_units=128)
  #   x = self.test_data, encoder.initial_state(2)
  #   outputs, h, c = encoder(x, training=True)
  #   self.assertEqual(outputs.shape, (self.batch_size, 5, self.n_units))
  #   self.assertEqual(h.shape, (self.batch_size, self.n_units))
  #   self.assertEqual(c.shape, (self.batch_size, self.n_units))

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

  def test_query_tiling_and_tensordot_for_mha(self):
    query = np.random.rand(self.batch_size, self.n_units).astype(np.float32)
    keys = np.random.rand(self.batch_size, self.max_seq,
                          self.n_units).astype(np.float32)
    values = keys

    n_head = 2
    tiled_query = tf.tile(query, [1, n_head])
    self.assertEqual(tiled_query.shape, (self.batch_size, self.n_units * 2))
    query_with_heads = tf.reshape(tiled_query,
                                  shape=(self.batch_size, n_head, self.n_units))
    for i in range(self.n_units):
      self.assertEqual(query_with_heads[0][0][i].numpy(),
                       query_with_heads[0][1][i].numpy())
    dense_layer = tf.keras.layers.Dense(self.n_units)

    a = np.random.rand(5, 3)
    b = np.random.rand(3, 5)
    self.assertEqual(tf.tensordot(a, b, axes=[[1], [0]]).shape, (5, 5))

    a = np.random.rand(5, 3)
    b = np.random.rand(2, 3, 5)
    self.assertEqual(tf.tensordot(a, b, axes=[[1], [1]]).shape, (5, 2, 5))
    a = np.random.rand(1, 5, 3)
    b = np.random.rand(2, 3, 5)
    self.assertEqual(tf.tensordot(a, b, axes=[[2], [1]]).shape, (1, 5, 2, 5))

    a = np.random.rand(self.batch_size, 1, self.n_units)
    b = np.random.rand(self.batch_size, self.max_seq, self.n_units)
    b = tf.transpose(b, [0, 2, 1])
    self.assertEqual(b.shape, (self.batch_size, self.n_units, self.max_seq))

    self.assertEqual(
        tf.tensordot(a, b, axes=[[2], [1]]).shape,
        (self.batch_size, 1, self.batch_size, self.max_seq))

    query_with_heads_w = dense_layer(query_with_heads)
    self.assertEqual(query_with_heads_w.shape,
                     (self.batch_size, n_head, self.n_units))

    query_with_heads_w_transposed = tf.transpose(query_with_heads_w,
                                                 perm=[0, 2, 1])
    self.assertEqual(query_with_heads_w_transposed.shape,
                     (self.batch_size, self.n_units, n_head))
    attention_score = tf.tensordot(query_with_heads_w_transposed,
                                   keys,
                                   axes=[[1], [2]])
    self.assertEqual(attention_score.shape,
                     (self.batch_size, n_head, self.batch_size, self.max_seq))

  # for i in range(n_head):
  #   tf.matmul(query, dense_layer(keys))

  # print(tf.tensordot(query_with_heads, dense_layer(keys)))
  # tf.do




if __name__ == '__main__':
  unittest.main()
