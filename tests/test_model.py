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
    self.q = tf.constant([[[0, 10, 0]]], dtype=tf.float32)  # (1, 3)
    self.k = tf.constant([[[10, 0, 0],
                           [0, 10, 0],
                           [0, 0, 10],
                           [0, 0, 10]]],
                         dtype=tf.float32)  # (4, 3)
    self.v = tf.constant([[[1, 0, 0],
                           [10, 0, 0],
                           [100, 5, 0],
                           [1000, 6, 0]]], dtype=tf.float32)  # (4, 3)
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

  def test_encoder(self):
    encoder = Encoder(vocab_size=10, n_units=128)
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

  def test_look_head_mask(self):

    x = tf.random.uniform(shape=(self.batch_size, self.max_seq),
                          minval=0,
                          maxval=self.vocab_size - 1,
                          dtype=tf.int32)
    x = self.x_for_mask
    mask = TransformerDecoder.create_look_head_mask(x, self.max_seq)
    # print(mask)

  # [[[0, 1, 1, 1, 1],
  #   [0, 0, 1, 1, 1],
  #   [0, 0, 0, 1, 1],
  #   [0, 0, 0, 0, 1],
  #   [0, 0, 0, 0, 0]],
  #  [[0, 1, 1, 1, 1],
  #   [0, 0, 1, 1, 1],
  #   [0, 0, 0, 1, 1],
  #   [0, 0, 0, 0, 1],
  #   [0, 0, 0, 0, 0]]]

  def test_scaled_dot_product_attention(self):
    scaled_dot_product_attention = ScaledDotProductAttention()
    seq_q = 5
    seq_kv = 6
    d_model = 25
    q, k, v = self.q, self.k, self.v
    batch_size = 1
    seq_q = 1
    seq_k = 4
    seq_v = 4

    c, aw = scaled_dot_product_attention(q, k, v, pad_mask=None)
    aw_list = list(aw.numpy()[0][0].astype(np.uint8))
    self.assertListEqual(aw_list, [0., 1., 0., 0.])
    self.assertEqual(c.shape, (batch_size, seq_q, 3))
    self.assertEqual(aw.shape, (batch_size, seq_q, seq_v))
    aw_ = np.sum(aw[0][0].numpy())
    self.assertEqual(aw_, 1.)

  def test_multi_head_attention(self):

    n_head = 5
    batch_size = 1
    seq_q = 1
    seq_k = 4
    seq_v = 4
    q, k, v = self.q, self.k, self.v
    d_model = q.shape[2]    # 3
    mha = MultiHeadAttention(n_head, d_model)
    c, aw = mha(q, k, v, mask=None)
    self.assertEqual(c.shape, (batch_size, seq_q, d_model))
    self.assertEqual(np.array(aw).shape, (n_head, batch_size, seq_q, seq_v))
    self.assertAlmostEqual(np.sum(np.array(aw)[0][0][0]), 1.)

  def test_encoder(self):
    x = np.random.randint(0, 100, size=(self.batch_size, self.max_seq))
    n_head = 8
    n_layer = 6
    encoder = TransformerEncoder(100, 100, 100, n_layer, n_head, 200)
    xs = encoder(x, attention_mask=encoder.create_pad_mask(x, 0), training=True)
    self.assertEqual(
        np.array(xs).shape, (n_layer, self.batch_size, self.max_seq, 100))

  def test_pad_mask(self):
    n_layer = 6
    n_head = 8

    x = self.x_for_mask

    batch_size = 3
    seq_len = 5

    d_ff = 2048
    te = TransformerEncoder(vocab_size=self.vocab_size,
                            d_model=self.n_units,
                            n_head=n_head,
                            n_layer=n_layer,
                            d_ff=d_ff,
                            learned_pos_enc=True,
                            seq_len=seq_len)

    self.assertEqual(
        tf.convert_to_tensor(
            te(x, attention_mask=te.create_pad_mask(x, 0),
               training=True)).shape,
        (n_layer, batch_size, seq_len, self.n_units))

    mask = te.create_pad_mask(x)
    self.assertEqual(mask.shape, (batch_size, seq_len, seq_len))
    self.assertListEqual(list(mask.numpy()[0][0]), [0., 0., 0., 0., 1.])

    temp_score = tf.ones(shape=(batch_size, seq_len, seq_len))

    mask *= -10e9
    # mask = mask[:, tf.newaxis, tf.newaxis, :]
    temp_score_masked = temp_score + mask
    self.assertEqual(temp_score_masked.shape, (batch_size, seq_len, seq_len))
    # print(temp_score_masked.numpy().astype(np.uint8))
    #     tf.Tensor(
    # [[[1. 1. 1. 1. 0.]
    #   [1. 1. 1. 1. 0.]
    #   [1. 1. 1. 1. 0.]
    #   [1. 1. 1. 1. 0.]
    #   [0. 0. 0. 0. 0.]]
    #
    #  [[1. 1. 1. 1. 0.]
    #   [1. 1. 1. 1. 0.]
    #   [1. 1. 1. 1. 0.]
    #   [1. 1. 1. 1. 0.]
    #   [0. 0. 0. 0. 0.]]
    #
    #  [[1. 1. 0. 0. 0.]
    #   [1. 1. 0. 0. 0.]
    #   [0. 0. 0. 0. 0.]
    #   [0. 0. 0. 0. 0.]
    #   [0. 0. 0. 0. 0.]]], shape=(3, 5, 5), dtype=float32)

    # Decoder's Pad masks
    q = tf.constant([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 1, 0],
                     [1, 2, 3, 4, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0],
                     [1, 2, 0, 0, 0, 0, 0]])    # (5, 7)
    # yapf: disable
    k = tf.constant([[1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 0],
                     [1, 2, 3, 0, 0],
                     [0, 0, 0, 0, 0]])    # (5, 5)
    # yapf: enable
    mask = TransformerDecoder.create_pad_mask(q, k, 0)
    print(mask)
    self.assertEqual(mask.shape, (5, 7, 5))

  def test_position_embedding(self):
    pe = PositionalEmbedding(self.n_units, self.vocab_size, True, self.max_seq)
    self.assertEqual(pe.pos_enc in pe.trainable_variables, True)

  def test_decode_layer(self):
    n_layer = 6
    n_head = 8
    vocab_size = self.vocab_size
    d_model = self.n_units
    seq_len = 5
    decoder = TransformerDecoderLayer(vocab_size, d_model, n_layer, n_head,
                                      d_model * 2, seq_len)

    batch_size = 2
    seq_len_input = 2
    inputs = tf.random.normal(shape=(batch_size, seq_len_input, d_model))
    q = np.random.randint(0, vocab_size, size=(batch_size, seq_len_input))
    k = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    outputs_encoder = tf.random.normal(shape=(batch_size, seq_len, d_model))
    attention_mask = TransformerDecoder.create_pad_mask(q, k, pad_idx=0)
    self.assertEqual(attention_mask.shape, (batch_size, seq_len_input, seq_len))
    self.assertEqual(inputs.shape, (batch_size, seq_len_input, d_model))
    self.assertEqual(outputs_encoder.shape, (batch_size, seq_len, d_model))
    self_attention_mask = TransformerDecoder.create_pad_mask(q, q, 0)
    self.assertEqual(self_attention_mask.shape,
                     (batch_size, seq_len_input, seq_len_input))

    x = decoder(inputs,
                outputs_encoder,
                attention_mask=attention_mask,
                training=True,
                self_attention_mask=self_attention_mask)
    self.assertEqual(x.shape, (batch_size, seq_len_input, d_model))

  def test_decoder(self):
    n_layer = 6
    d_model = 128
    n_head = 8
    vocab_size = self.vocab_size
    d_ff = 2048
    seq_len = self.max_seq
    learned_pos_enc = False
    batch_size = self.batch_size
    decoder = TransformerDecoder(vocab_size, n_layer, d_model, n_head, d_ff,
                                 seq_len, learned_pos_enc)
    data_loader = DataLoader('../data/input/aihub_kor-eng/1.구어체.xlsx',
                             n_data=10)
    tokenizer: tf.keras.preprocessing.text.Tokenizer = data_loader.tokenizer.kor
    pad_idx = tokenizer.word_index[tokenizer.oov_token]
    self.assertEqual(tokenizer.oov_token, '<unk>')
    self.assertEqual(pad_idx, 1)

    start_tokens = tf.constant(
        tokenizer.texts_to_sequences(['<start>'] * batch_size))
    outputs_encoder = tf.random.normal(shape=(n_layer, batch_size, seq_len,
                                              d_model),
                                       dtype=tf.float32)
    k = tf.random.uniform(shape=(batch_size, seq_len),
                          dtype=tf.int32,
                          minval=0,
                          maxval=10)
    attention_mask = TransformerDecoder.create_pad_mask(start_tokens, k, 0)
    self_attention_mask = TransformerDecoder.create_pad_mask(
        start_tokens, start_tokens, 0)
    logits = decoder(start_tokens,
                     outputs_encoder,
                     training=True,
                     attention_mask=attention_mask,
                     self_attention_mask=self_attention_mask)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
    # inputs = tf.random.uniform(shape=(batch_size, seq_len))


if __name__ == '__main__':
  unittest.main()
