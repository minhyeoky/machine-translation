import numpy as np
import tensorflow as tf

INF = 10e14

keras = tf.keras
LSTM = keras.layers.LSTM

Model = keras.models.Model
Layer = keras.layers.Layer
Dense = keras.layers.Dense
LayerNorm = keras.layers.LayerNormalization
Embedding = keras.layers.Embedding

# https://www.tensorflow.org/tutorials/text/transformer
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#
#   def __init__(self, d_model, warmup_steps=4000):
#     super(CustomSchedule, self).__init__()
#
#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)
#
#     self.warmup_steps = warmup_steps
#
#   def __call__(self, step):
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps**-1.5)
#
#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class PositionalWiseFeedForward(Layer):

  def __init__(self, d_ff, d_model):
    super(PositionalWiseFeedForward, self).__init__()
    self.W1 = Dense(units=d_ff, activation='relu')
    self.W2 = Dense(units=d_model, activation=None)

  def call(self, inputs, **kwargs):
    """

    Args:
      inputs: `(batch_size, seq_pos, d_model)`
      **kwargs:

    Returns:
      outputs: `(batch_size, seq_pos, d_model)`
    """

    x = self.W1(inputs)
    x = self.W2(x)

    return x


class ScaledDotProductAttention(Layer):

  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()

  def call(self, q, k, v, pad_mask=None, look_ahead_mask=None, **kwargs):
    """

    Args:
      q: queries, `(batch_size, seq_q, d_model)`
      k: keys, `(batch_size, seq_k, d_model)`
      v: values `(batch_size, seq_v, d_model)`
      **kwargs:

    Returns:
      context_vector: `(batch_size, seq_q, d_model)`
      attention_weight: `(batch_size, seq_q, seq_v)`
    """
    # `(batch_size, seq_q, seq_k)`
    d_model = tf.cast(q.shape[2], tf.float32)
    score = tf.matmul(q, k, transpose_b=True)
    score_logits = tf.divide(score, tf.sqrt(d_model))

    if pad_mask is not None:
      pad_mask = pad_mask * -INF
      # masking: -inf 로 만들어서 softmax 영향력이 없도록 함
      score_logits = score_logits + pad_mask

    if look_ahead_mask is not None:
      look_ahead_mask = look_ahead_mask * -INF
      score_logits = score_logits + look_ahead_mask

    # softmax function on the seq_k.
    # shape == `(batch_size, seq_q, seq_k)`
    attention_weight = tf.math.softmax(score_logits, axis=-1)

    # context vector shape == `(batch_size, seq_q, d_model)`
    # weighted sum
    context_vector = tf.matmul(attention_weight, v)

    return context_vector, attention_weight


class MultiHeadAttention(Layer):

  def __init__(self, n_head, d_model, *args, **kwargs):
    super(MultiHeadAttention, self).__init__(*args, **kwargs)

    self.d_model = d_model
    self.n_head = n_head

    self.Wq = Dense(units=d_model * n_head, activation=None)
    self.Wk = Dense(units=d_model * n_head, activation=None)
    self.Wv = Dense(units=d_model * n_head, activation=None)
    self.Wo = Dense(units=d_model, activation=None)
    self.scaled_dot_product_attention = ScaledDotProductAttention()

  def __call__(self,
               query,
               keys,
               values,
               pad_mask=None,
               look_ahead_mask=None,
               **kwargs):
    """

    Args:
      query: `(batch_size, seq_q, d_model)`
      keys: `(batch_size, seq_k, d_model)`
      values: `(batch_size, seq_v, d_model)`
      pad_mask: `(batch_size, seq_q, seq_k)`
      look_ahead_mask: `(batch_size, seq_q, seq_k)`

    Returns:
      context_vectors: `(batch_size, seq_q, d_model)`
      attention_weights: `(n_head, batch_size, seq_q, seq_v)`
    """

    # query shape == (batch_size, seq_q * d_model)
    batch_size = query.shape[0]
    seq_q = query.shape[1]
    seq_k = keys.shape[1]
    seq_v = values.shape[1]

    # query_with_heads, shape == `(batch_size, n_head, seq_q, d_model)`
    query_with_heads = tf.reshape(self.Wq(query),
                                  shape=(batch_size, seq_q, self.n_head,
                                         self.d_model))
    query_with_heads = tf.transpose(query_with_heads, perm=[0, 2, 1, 3])

    keys_with_heads = tf.reshape(self.Wk(keys),
                                 shape=(batch_size, seq_k, self.n_head,
                                        self.d_model))
    keys_with_heads = tf.transpose(keys_with_heads, perm=[0, 2, 1, 3])
    values_with_heads = tf.reshape(self.Wv(values),
                                   shape=(batch_size, seq_v, self.n_head,
                                          self.d_model))
    values_with_heads = tf.transpose(values_with_heads, perm=[0, 2, 1, 3])

    # context_vectors shape == `(n_head, batch_size, seq_q, d_model)`
    context_vectors = []
    attention_weights = []
    for h in range(self.n_head):
      q = query_with_heads[:, h, :]
      k = keys_with_heads[:, h, :]
      v = values_with_heads[:, h, :]

      q = tf.reshape(q, shape=(batch_size, seq_q, self.d_model))
      k = tf.reshape(k, shape=(batch_size, seq_k, self.d_model))
      v = tf.reshape(v, shape=(batch_size, seq_v, self.d_model))

      context_vector, attention_weight = self.scaled_dot_product_attention(
          q, k, v, pad_mask=pad_mask, look_ahead_mask=look_ahead_mask)
      attention_weights.append(attention_weight)
      context_vectors.append(context_vector)

    # concatnate heads, shape == `(batch_size, seq_q, d_model * n_head)`
    context_vectors = tf.concat(context_vectors, axis=-1)
    assert context_vectors.shape == (batch_size, seq_q,
                                     self.d_model * self.n_head)

    # output context vector shape == `(batch_size, seq_q, d_model)`
    context_vectors = self.Wo(context_vectors)

    return context_vectors, attention_weights


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(
      np.arange(position)[:, np.newaxis],
      np.arange(d_model)[np.newaxis, :], d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class Encoder(Model):

  def __init__(self,
               vocab_size,
               d_model,
               n_layer,
               n_head,
               d_ff,
               learned_pos_enc=False,
               seq_len=None):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.n_layer = n_layer
    self.learned_pos_enc = learned_pos_enc

    self.positional_embedding = PositionalEmbedding(
        d_model, vocab_size, learned_pos_enc=learned_pos_enc, seq_len=seq_len)
    # sub-layer 1: multi head attention
    # sub-layer 2: positional feed forward
    self.encoder_layers = [[
        MultiHeadAttention(n_head, d_model),
        PositionalWiseFeedForward(d_ff, d_model),
        LayerNorm(),
        LayerNorm()
    ] for _ in range(n_layer)]

  def call(self, inputs, training, attention_mask, **kwargs):
    """

    Args:
      inputs: `(batch_size, seq_len)`
      **kwargs:

    Returns:
      x_s: `(n_layer, batch_size, seq_len, d_model)`
        outputs of layers

    """

    # Embedding
    seq_len = inputs.shape[1]
    x = self.positional_embedding(inputs, seq_len)

    # Add Positional Encoding

    x_s = []
    for i in range(self.n_layer):
      prev_x = x
      multi_head_attention = self.encoder_layers[i][0]
      pos_wise_forward = self.encoder_layers[i][1]
      layer_norm_1 = self.encoder_layers[i][2]
      layer_norm_2 = self.encoder_layers[i][3]
      x, attention_weights = multi_head_attention(x, x, x, pad_mask=attention_mask)
      x = layer_norm_1(prev_x + x, training=training)
      prev_x = x

      x = pos_wise_forward(x)
      x = layer_norm_2(prev_x + x, training=training)
      x_s.append(x)

    return x_s

  @staticmethod
  def create_pad_mask(x, pad_idx=0):
    """PAD -> Mask"""
    mask = tf.cast(tf.math.equal(x, pad_idx), dtype=tf.float32)
    seq_len = mask.shape[1]
    batch_size = x.shape[0]
    mask_row = tf.reshape(mask, shape=(batch_size, 1, seq_len))
    mask_col = tf.reshape(mask, shape=(batch_size, seq_len, 1))
    mask = mask_col + mask_row
    return mask


class DecoderLayer(Layer):

  def __init__(self,
               vocab_size,
               d_model,
               n_head,
               d_ff,
               seq_len=None,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.d_model = d_model
    self.seq_len = seq_len

    # Sub layers
    self.masked_multi_head_attention = MultiHeadAttention(n_head, d_model)
    self.multi_head_attention = MultiHeadAttention(n_head, d_model)
    self.positional_feed_forward = PositionalWiseFeedForward(d_ff, d_model)
    self.layer_norm = (LayerNorm(), LayerNorm(), LayerNorm())

  def call(self,
           inputs,
           outputs_encoder,
           training,
           attention_mask=None,
           look_ahead_mask=None,
           self_attention_mask=None,
           **kwargs):
    """

    Args:
      inputs: `(batch_size, seq_len, d_model)`
        Embedded & Position encoded input vector
      outputs_encoder: `(batch_size, encoder_seq_len, d_model)`
      training:


    Returns:
      logits: `(batch_size, seq_len, d_model)`


    """

    # Decoder Layers
    keys = values = outputs_encoder
    masked_multi_head_attention = self.masked_multi_head_attention
    multi_head_attention = self.multi_head_attention
    feed_forward = self.positional_feed_forward
    layer_norm = self.layer_norm

    # 1
    prev_x = x = inputs
    x, attention_weights_1 = masked_multi_head_attention(
        query=x,
        keys=x,
        values=x,
        pad_mask=self_attention_mask,
        look_ahead_mask=look_ahead_mask)
    x = layer_norm[0](x + prev_x, training=training)

    # 2
    prev_x = x
    x, attention_weights_2 = multi_head_attention(x,
                                                  keys,
                                                  values,
                                                  pad_mask=attention_mask,
                                                  look_ahead_mask=None)
    x = layer_norm[1](x + prev_x, training=training)

    # 3
    prev_x = x
    x = feed_forward(x)
    x = layer_norm[2](x + prev_x, training=training)

    return x


class PositionalEmbedding(Layer):

  def __init__(self, d_model, vocab_size, learned_pos_enc=False, seq_len=None):
    super(PositionalEmbedding, self).__init__()
    self.embedding = Embedding(vocab_size, d_model)
    self.learned_pos_enc = learned_pos_enc
    self.d_model = d_model
    self.pos_enc = None
    if seq_len and learned_pos_enc:
      # self.pos_enc = tf.Variable(trainable=None,
      #                           initial_value=tf.random.normal(
      #                               (1, seq_len, d_model), dtype=tf.float32),
      #                           dtype=tf.float32)
      self.pos_enc = self.add_weight(name='pos_enc',
                                     shape=(1, seq_len, d_model),
                                     dtype=tf.float32,
                                     trainable=True,
                                     initializer=tf.initializers.glorot_normal)

  def call(self, inputs, seq_len):
    """

    Args:
      inputs: `(batch_size, seq_len)`
      seq_len: position

    Returns:
      outputs: `(batch_size, seq_len, d_model)`
    """
    if not self.learned_pos_enc:
      self.pos_enc = positional_encoding(seq_len, self.d_model)
    inputs = self.embedding(inputs)
    embedded_inputs = inputs + self.pos_enc
    return embedded_inputs


class Decoder(Model):

  def __init__(self,
               vocab_size,
               n_layer,
               d_model,
               n_head,
               d_ff,
               seq_len=None,
               learned_pos_enc=False):
    super(Decoder, self).__init__()
    self.n_head = n_head
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.n_layer = n_layer

    self.positional_embedding = PositionalEmbedding(d_model, vocab_size,
                                                    learned_pos_enc, seq_len)

    self.decode_layers = [
        DecoderLayer(vocab_size, d_model, n_head, d_ff, seq_len)
        for _ in range(n_layer)
    ]
    self.logit_layer = Dense(units=vocab_size, activation=None)

  def call(self,
           inputs,
           outputs_encoder,
           training,
           attention_mask=None,
           look_ahead_mask=None,
           self_attention_mask=None):
    """

    Args:
      inputs: `(batch_size, seq_len)`
      outputs_encoder: `(n_layer, batch_size, seq_len_encoder, d_model)`

    Returns:
      logits: `(batch_size, seq_len, vocab_size)`
    """

    # Positional Embedding
    seq_len = inputs.shape[1]
    x = self.positional_embedding(inputs, seq_len)

    for i in range(self.n_layer):
      decode_layer = self.decode_layers[i]
      outputs_encoder_i_ = outputs_encoder[i]
      x = decode_layer(x,
                       outputs_encoder_i_,
                       attention_mask=attention_mask,
                       look_ahead_mask=look_ahead_mask,
                       self_attention_mask=self_attention_mask,
                       training=training)

    return self.logit_layer(x)

  @staticmethod
  def create_pad_mask(q, k, pad_idx: int):
    """

    Args:
      q: queries, `(batch_size, seq_q)`
      k: keys, `(batch_size, seq_k)`

    Returns:
      mask: `(batch_size, seq_q, seq_k)`
    """
    mask_q = tf.cast(tf.equal(q, pad_idx), dtype=tf.float32)[:, :, tf.newaxis]
    mask_k = tf.cast(tf.equal(k, pad_idx), dtype=tf.float32)[:, tf.newaxis, :]
    mask = mask_q + mask_k
    return mask

  @staticmethod
  def create_look_head_mask(x, seq_len):
    # mask = tf.ones(shape=(seq_len, seq_len), dtype=tf.float32)

    mask = []
    for i in range(seq_len):
      mask.append([0] * (i + 1) + [1] * (seq_len - i - 1))
      # mask[i][] *= 0
    batch_size = x.shape[0]
    mask = tf.constant([mask] * batch_size,
                       dtype=tf.float32,
                       shape=(batch_size, seq_len, seq_len))

    return mask
