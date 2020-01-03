import tensorflow as tf
from .seq2seq import Decoder

keras = tf.keras
LSTM = keras.layers.LSTM
Decoder = Decoder


class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_size, n_units):
        super(Encoder, self).__init__()

        self.n_units = n_units

        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_size
        )

        # Bidirectional Wrapper
        # 인스턴스를 받고, 그 인스턴스를 복제 -> 복제 된 인스턴스는 입력값을 반대로 받음
        # Receives an instance, replicate, The replicated instance receives input values reversed
        self.lstm = keras.layers.Bidirectional(
            layer=LSTM(units=n_units, return_sequences=True, return_state=True),
            merge_mode="concat",
            backward_layer=None,
        )

    def call(self, inputs, **kwargs):
        """

    Args:
      inputs:
        x: token vectors with shape `(batch_size, max_length)`
        initial_state: states for cell
      **kwargs:

    Returns:
      ret:
        outputs: concatenated output with shape `(batch_size, max_length, n_units * 2)`
        h_f, h_b, c_f, c_b.
    """
        x, initial_state, = inputs
        training = kwargs.get("training", True)

        x = self.embedding(x)

        if training:
            return self.lstm(x, initial_state=initial_state)
        else:
            return self.lstm(x, initial_state=initial_state)

    def initial_state(self, batch_size):
        return [tf.zeros((batch_size, self.n_units))] * 4  # [h, c]
