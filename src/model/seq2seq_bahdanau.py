import tensorflow as tf

keras = tf.keras
LSTM = keras.layers.LSTM
Layer = tf.keras.layers.Layer

Model = keras.Model


class Encoder(Model):

  def __init__(self, vocab_size, embedding_size, n_units):
    super(Encoder, self).__init__()

    self.n_units = n_units

    self.embedding = keras.layers.Embedding(input_dim=vocab_size,
                                            output_dim=embedding_size)

    self.lstm = keras.layers.LSTM(units=n_units,
                                  activation='tanh',
                                  recurrent_activation='sigmoid',
                                  return_sequences=True,
                                  return_state=True)

  def call(self, inputs, **kwargs):
    """

    Args:
      inputs:
        x: token tensor with shape `(batch_size, max_length)`
        initial_state: state tensors for cell
          h: hidden state tensor with shape `(batch_size, n_units)`
          c: cell state tensor with shape `(batch_size, n_units)`
      **kwargs:

    Returns:
      `(outputs, h, c)`
        outputs: tensor with shape `(batch_size, max_length, n_units)`
        h: hidden state tensor with shape `(batch_size, n_units)`
        c: cell state tensor with shape `(batch_size, n_units)`

    """
    x, initial_state, = inputs
    training = kwargs.get('training', True)
    x = self.embedding(x)

    if training:
      outputs, h, c = self.lstm(x, initial_state=initial_state)
      return outputs, h, c
    else:
      outputs, h, c = self.lstm(x, initial_state=initial_state)
      return outputs, h, c

  def initial_state(self, batch_size):
    return [tf.zeros((batch_size, self.n_units))] * 2    # [h, c]


class Decoder(Model):

  def __init__(self, vocab_size, embedding_size, n_units, bahdanau_units):
    super(Decoder, self).__init__()

    self.embedding = keras.layers.Embedding(input_dim=vocab_size,
                                            output_dim=embedding_size)
    self.lstm = keras.layers.LSTM(units=n_units,
                                  activation='tanh',
                                  recurrent_activation='sigmoid',
                                  return_sequences=True,
                                  return_state=True)
    self.dense = keras.layers.Dense(units=vocab_size, activation=None)

    self.bahdanau_attention = BahdanauAttention(units=bahdanau_units)

  def call(self, inputs, **kwargs):
    """

    Args:
      inputs:
        x: input vector for cell with shape [batch_size, 1]
        values: list of output tensor of encoder with shape [batch_size, max_length, n_units]
        initial_state: hidden states, cell state
      **kwargs:

    Returns:
      outputs: output vectors with shape [batch_size, vocab_size]
      h: hidden state vectors with shape [batch_size, n_units]
      c: cell state vectors with shape [batch_size, n_units]

    """
    x, values, initial_state = inputs
    query = tf.concat(initial_state, 1)
    context_vector, attention_weights = self.bahdanau_attention(inputs=(query,
                                                                        values))
    x = self.embedding(x)

    # x shape == `(batch_size, 1, embedding_size + n_units * 2)`
    context_vector = tf.expand_dims(context_vector, axis=1)
    x = tf.concat([x, context_vector], axis=-1)

    outputs, h, c = self.lstm(x, initial_state=initial_state)
    outputs = tf.reshape(outputs, shape=(x.shape[0], outputs.shape[2]))
    outputs = self.dense(outputs)
    return outputs, h, c


class BahdanauAttention(Layer):

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, inputs, **kwargs):
    """
      Args:
        inputs:
          q: decoder's target hidden state tensor with shape `(batch_size, n_units * 2)`
          values: all of the hidden states tensor of encoder equals to key
            with shape `(batch_size, max_length, n_units * 2)`

      Returns:
        context_vector: tensor of weighted sum of values with shape `(batch_size, n_units * 2)`
        attention_weights: tensor of attention probabilities with shape `(batch_size, max_length)`
    """
    # hidden shape == (batch_size, n_units * 2)
    # hidden_with_time_axis shape == (batch_size, 1, n_units * 2)
    # we are doing this to perform addition to calculate the score
    query, values = inputs
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, n_units * 2)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, n_units * 2)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
