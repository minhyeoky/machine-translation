import tensorflow as tf

keras = tf.keras
LSTM = keras.layers.LSTM


class Encoder(keras.Model):

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
        x, initial_state, = inputs
        training = kwargs.get('training', True)

        x = self.embedding(x)

        if training:
            outputs, h, c = self.lstm(x, initial_state=initial_state)

            return outputs, h, c

    def initial_state(self, batch_size):
        return [tf.zeros((batch_size, self.n_units))] * 2  # [h, c]


class Decoder(keras.Model):

    def __init__(self, vocab_size=None, embedding_size=None, n_units=None):
        super(Decoder, self).__init__()

        self.embedding = keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim=embedding_size)
        self.lstm = keras.layers.LSTM(units=n_units,
                                      activation='tanh',
                                      recurrent_activation='sigmoid',
                                      return_sequences=True,
                                      return_state=True)
        self.dense = keras.layers.Dense(units=vocab_size,
                                        activation=None)

    def call(self, inputs, **kwargs):
        # x - [batch_size, 1]
        x, initial_state = inputs
        x = self.embedding(x)

        outputs, h, c = self.lstm(x, initial_state=initial_state)
        outputs = tf.reshape(outputs, shape=(x.shape[0], outputs.shape[2]))
        outputs = self.dense(outputs)
        return outputs, h, c

# class BahdanauAttention(tf.keras.layers.Layer):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def call(self, inputs, **kwargs):
#
#         pass
