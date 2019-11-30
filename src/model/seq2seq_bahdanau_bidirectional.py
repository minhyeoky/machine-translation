import tensorflow as tf
from .seq2seq_bidirectional import Encoder
from .seq2seq_bahdanau import Decoder
keras = tf.keras
LSTM = keras.layers.LSTM
Layer = tf.keras.layers.Layer

Model = keras.Model
Encoder = Encoder
Decoder = Decoder

