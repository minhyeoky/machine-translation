import numpy as np
import tensorflow as tf

INF = 10e9

keras = tf.keras
LSTM = keras.layers.LSTM

Model = keras.models.Model
Layer = keras.layers.Layer
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
LayerNorm = keras.layers.LayerNormalization
Embedding = keras.layers.Embedding


class PositionWiseFeedFoward(Layer):
    def __init__(self, d_ff, d_model, dropout_rate):
        super(PositionWiseFeedFoward, self).__init__()
        self.W1 = Dense(units=d_ff, use_bias=True, activation="relu")
        self.W2 = Dense(units=d_model, use_bias=True, activation=None)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x, padding=None, training=True):
        """

        Args:
            x: `(batch_size, seq_len, d_model)`

        Returns:
            outputs: `(batch_size, seq_len, d_model)`
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        d_model = x.shape[2]

        if padding is not None:
            pad_mask = tf.reshape(padding, shape=[-1], name="pad_mask")
            nonpad_ids = tf.cast(tf.where(pad_mask == 1), dtype=tf.int32)

            #
            x = tf.reshape(x, shape=[-1])
            x = tf.gather_nd(
                x, indices=nonpad_ids, batch_dims=0, name="exclude_padding"
            )
            x = tf.expand_dims(x, axis=0)
            assert x.shape == (1, nonpad_ids.shape[0], d_model)

        x = self.W1(x)
        x = self.dropout(x, training=training)
        x = self.W2(x)

        if padding is not None:

            x = tf.squeeze(x, axis=0)
            x = tf.scatter_nd(
                indices=nonpad_ids, updates=x, shape=(batch_size * seq_len, d_model)
            )
            x = tf.reshape(x, shape=[batch_size, seq_len, d_model])

        return x


class ScaledDotProductAttention(Layer):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def call(self, q, k, v, mask):
        """

    Args:
      q: `(batch_size, n_head, seq_q, d_model)`
      k: `(batch_size, n_head, seq_k, d_model)`
      v: `(batch_size, n_head, seq_v, d_model)`
      In case of self attention, q = k = v
      mask: `(batch_size, 1, 1, seq_k)`: mask padding tokens.
      or `(batch_size, 1, seq_q, seq_k)`: to block inappropriate information and also mask padding tokens.

    Returns:
      context_vector: `(batch_size, n_head, seq_q, d_model)`
      attention_weights: `(batch_size, n_head, seq_q, seq_v)`
    """
        batch_size = q.shape[0]
        n_head = q.shape[1]
        seq_q, seq_k, seq_v = q.shape[2], k.shape[2], v.shape[2]
        d_model_q, d_model_k, d_model_v = q.shape[3], k.shape[3], v.shape[3]
        assert batch_size == k.shape[0] == v.shape[0]
        assert n_head == k.shape[1] == v.shape[1]

        score = tf.matmul(q, k, transpose_b=True)
        score_logits = tf.divide(score, tf.sqrt(tf.cast(d_model_q, dtype=tf.float32)))
        assert score_logits.shape == (batch_size, n_head, seq_q, seq_k)

        if mask is not None:
            score_logits = score_logits + (
                mask * -INF
            )  # masking: -inf 로 만들어서 softmax function 의 영향력이 없도록 함

        attention_weights = tf.math.softmax(score_logits, axis=-1)
        assert attention_weights.shape == (batch_size, n_head, seq_q, seq_k)
        assert v.shape == (batch_size, n_head, seq_k, d_model_v)

        # attention weights 각각의 행에는 value 들에 대한 score가 있고 각각에 대응하는 value 는 열벡터이므로, 자연스럽게 weighted sum 이 됨
        context_vector = tf.matmul(attention_weights, v)
        assert context_vector.shape == (batch_size, n_head, seq_q, d_model_v)

        return context_vector, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, n_head, d_model):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head

        # Linear transformation
        self.Wq = Dense(units=d_model, use_bias=False, activation=None)
        self.Wk = Dense(units=d_model, use_bias=False, activation=None)
        self.Wv = Dense(units=d_model, use_bias=False, activation=None)
        self.Wo = Dense(units=d_model, use_bias=False, activation=None)
        self.dh = d_model // n_head
        assert d_model % n_head == 0.0
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        x = tf.transpose(
            tf.reshape(x, shape=(batch_size, seq_len, self.n_head, self.dh)),
            perm=[0, 2, 1, 3],
        )
        assert x.shape == (batch_size, self.n_head, seq_len, self.dh)
        return x

    def call(self, q, k, v, mask):
        """

    Args:
      q: `(batch_size, seq_q, d_model)`
      k: `(batch_size, seq_k, d_model)`
      v: `(batch_size, seq_v, d_model)`
      mask: `(batch_size, 1, 1, seq_q)` or `(batch_size, 1, seq_q, seq_k)`

    Returns:
      context_vectors: `(batch_size, seq_q, d_model)`
      attention_weights: `(n_head, batch_size, seq_q, seq_v)`
    """
        batch_size = q.shape[0]
        seq_q, seq_k, seq_v = q.shape[1], k.shape[1], v.shape[1]
        assert batch_size == k.shape[0] == v.shape[0]

        # Linear projection
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # query_with_heads, shape == `(batch_size, n_head, seq_q, d_model)`
        query_with_heads = self.split_heads(q)
        keys_with_heads = self.split_heads(k)
        values_with_heads = self.split_heads(v)

        context_vector, attention_weights = self.scaled_dot_product_attention(
            query_with_heads, keys_with_heads, values_with_heads, mask=mask
        )

        assert context_vector.shape == (batch_size, self.n_head, seq_q, self.dh)
        context_vector = tf.transpose(context_vector, perm=[0, 2, 1, 3])
        assert context_vector.shape == (batch_size, seq_q, self.n_head, self.dh)
        context_vector = tf.reshape(
            context_vector, shape=(batch_size, -1, self.d_model)
        )
        context_vector = self.Wo(context_vector)
        assert context_vector.shape == (batch_size, seq_q, self.d_model)

        return context_vector, attention_weights


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    assert pos_encoding.shape == (1, position, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(Layer):
    def __init__(self, d_model, n_head, d_ff):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(n_head, d_model)
        self.position_wise_feed_forward = PositionWiseFeedFoward(d_ff, d_model)
        self.layer_norm_1 = LayerNorm()
        self.layer_norm_2 = LayerNorm()
        self.dropout = Dropout(rate=0.1)

    def call(self, x, mask, training):
        x = self.dropout(x)
        __x, _ = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm_1(x + __x, training=training)
        x = self.dropout(x)
        __x = self.position_wise_feed_forward(x)
        x = self.layer_norm_2(x + __x, training=training)
        return x


class Encoder(Model):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer

        self.pos_embed = PositionalEmbedding(d_model, vocab_size)
        self.encoder_layers = [
            EncoderLayer(d_model, n_head, d_ff) for _ in range(n_layer)
        ]
        self.dropout = Dropout(rate=0.1)

    def call(self, inputs, training, pad_mask, **kwargs):
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
        x = self.pos_embed(inputs, seq_len)
        x = self.dropout(x)

        for i in range(self.n_layer):
            x = self.encoder_layers[i](x, mask=pad_mask, training=training)

        return x  # (batch_size, seq_len, d_model)


class DecoderLayer(Layer):
    def __init__(
        self, vocab_size, d_model, n_head, d_ff, seq_len=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len

        # Sub layers
        self.masked_multi_head_attention = MultiHeadAttention(n_head, d_model)
        self.multi_head_attention = MultiHeadAttention(n_head, d_model)
        self.positional_feed_forward = PositionWiseFeedFoward(d_ff, d_model)
        self.layer_norm = (LayerNorm(), LayerNorm(), LayerNorm())
        self.dropout = Dropout(rate=0.1)

    def call(
        self,
        x,
        outputs_encoder,
        training,
        pad_mask=None,
        look_ahead_mask=None,
        **kwargs
    ):
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
        k = v = outputs_encoder

        # 1
        x = self.dropout(x)
        __x, attention_weights_1 = self.masked_multi_head_attention(
            x, x, x, mask=look_ahead_mask
        )
        x = self.layer_norm[0](x + __x, training=training)

        # 2
        x = self.dropout(x)
        __x, attention_weights_2 = self.multi_head_attention(x, k, v, mask=pad_mask)
        x = self.layer_norm[1](x + __x, training=training)

        # 3
        x = self.dropout(x)
        __x = self.positional_feed_forward(x)
        x = self.layer_norm[2](x + __x, training=training)

        return x


class PositionalEmbedding(Layer):
    def __init__(self, d_model, vocab_size):
        super(PositionalEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, inputs, seq_len):
        """

    Args:
      inputs: `(batch_size, seq_len)`
      seq_len: position

    Returns:
      outputs: `(batch_size, seq_len, d_model)`
    """
        seq_len = inputs.shape[1]

        pos_enc = positional_encoding(1000, self.d_model)[:, :seq_len, :]
        embedded_inputs = self.embedding(inputs)
        embedded_inputs *= tf.sqrt(tf.cast(self.d_model, tf.float32))  # ??
        embedded_inputs = embedded_inputs + pos_enc
        assert embedded_inputs.shape == inputs.shape + (
            self.d_model,
        )  # (batch_size, seq_len, d_model)
        return embedded_inputs


class Decoder(Model):
    def __init__(
        self,
        vocab_size,
        n_layer,
        d_model,
        n_head,
        d_ff,
        seq_len=None,
        learned_pos_enc=False,
    ):
        super(Decoder, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layer = n_layer

        self.positional_embedding = PositionalEmbedding(d_model, vocab_size)

        self.decode_layers = [
            DecoderLayer(vocab_size, d_model, n_head, d_ff, seq_len)
            for _ in range(n_layer)
        ]
        self.logit_layer = Dense(units=vocab_size, activation=None)
        self.dropout = Dropout(rate=0.1)

    def call(self, inputs, outputs_encoder, training, pad_mask, look_ahead_mask):
        """

    Args:
      inputs: `(batch_size, seq_len)`
      outputs_encoder: `(batch_size, seq_len_encoder, d_model)`

    Returns:
      logits: `(batch_size, seq_len, d_model)`
    """

        # Positional Embedding
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        x = self.positional_embedding(inputs, seq_len)
        x = self.dropout(x)

        for i in range(self.n_layer):
            decode_layer = self.decode_layers[i]
            x = decode_layer(
                x,
                outputs_encoder,
                pad_mask=pad_mask,
                look_ahead_mask=look_ahead_mask,
                training=training,
            )
        assert x.shape == (batch_size, seq_len, self.d_model)
        return x


# def create_decoder_pad_mask(q, k, pad_idx: int):
#   """
#
#   Args:
#     q: queries, `(batch_size, seq_q)`
#     k: keys, `(batch_size, seq_k)`
#
#   Returns:
#     mask: `(batch_size, seq_q, seq_k)`
#   """
#   mask_q = tf.cast(tf.equal(q, pad_idx), dtype=tf.float32)[:, :, tf.newaxis]
#   mask_k = tf.cast(tf.equal(k, pad_idx), dtype=tf.float32)[:, tf.newaxis, :]
#   mask = mask_q + mask_k
#   return mask


class Transformer(tf.keras.Model):
    def __init__(
        self, n_layer, d_model, n_head, d_ff, input_vocab_size, target_vocab_size
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_layer=n_layer,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            vocab_size=input_vocab_size,
        )

        self.decoder = Decoder(
            n_layer=n_layer,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            vocab_size=target_vocab_size,
        )
        self.target_vocab_size = target_vocab_size

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self, inputs, training, enc_pad_mask, dec_pad_mask, look_ahead_mask, **kwargs
    ):
        """

    Args:

    Returns:
      outputs: `(batch_size, tar_seq_len, tar_vocab_size)`
      attention_weights: 

    """
        inp, tar = inputs

        enc_output = self.encoder(inp, training, enc_pad_mask)
        dec_output = self.decoder(
            tar, enc_output, training, dec_pad_mask, look_ahead_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)
        batch_size, tar_seq_len = inp.shape[0], tar.shape[1]
        assert final_output.shape == (batch_size, tar_seq_len, self.target_vocab_size)
        return final_output
