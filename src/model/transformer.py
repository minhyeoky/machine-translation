import tensorflow as tf

from src.model.transformer_utils import positional_encoding

MAX_POSITION = 10000

INF = 1e7

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
        q: `(..., seq_q, d_model)`
        k: `(..., seq_k, d_model)`
        v: `(..., seq_v, d_model)`
            In case of self attention, q = k = v
            key == value
        mask:
            `(..., 1, seq_k)`: padding mask, broadcastable.
            or `(..., seq_q, seq_k)`: to block inappropriate information and also mask padding tokens.

        Returns:
         context_vector: `(..., seq_k, d_model)`
         attention_weights: `(..., seq_q, seq_k)`
        """
        seq_q, seq_k, seq_v = q.shape[-2], k.shape[-2], v.shape[-2]
        d_model_q, d_model_k, d_model_v = q.shape[-1], k.shape[-1], v.shape[-1]

        # dot product between query's last axis and key's last axis
        score = tf.matmul(q, k, transpose_b=True)
        # scaling according to key's dimension size
        score_logits = tf.divide(score, tf.sqrt(tf.cast(d_model_k, dtype=tf.float32)))

        if mask is not None:
            # masking: -inf 로 만들어서 softmax function 의 영향력이 없도록 함
            score_logits = score_logits + mask * -INF

        # TODO apply dropout to attention weights (official tensorflow transformer)
        # https://www.groundai.com/project/dropattention-a-regularization-method-for-fully-connected-self-attention-networks/1
        attention_weights = tf.math.softmax(score_logits, axis=-1)

        assert attention_weights.shape[-2:] == (seq_q, seq_k)
        assert v.shape[-2:] == (seq_k, d_model_v)

        # Weighted sum.
        context_vector = tf.matmul(attention_weights, v)

        assert context_vector.shape[-2:] == (seq_v,)
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

        Returns:
            context_vectors: `(batch_size, seq_q, d_model)`
            attention_weights: `(n_head, batch_size, seq_q, seq_v)`
        """
        batch_size = q.shape[0]
        seq_q, seq_k, seq_v = q.shape[1], k.shape[1], v.shape[1]

        # Project linearly
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        query_with_heads = self.split_heads(q)
        keys_with_heads = self.split_heads(k)
        values_with_heads = self.split_heads(v)

        context_vector, attention_weights = self.scaled_dot_product_attention(
            query_with_heads, keys_with_heads, values_with_heads, mask=mask
        )

        # Check shapes of heads
        assert context_vector.shape == (batch_size, self.n_head, seq_q, self.dh)

        # Combine heads
        context_vector = tf.transpose(context_vector, perm=[0, 2, 1, 3])
        assert context_vector.shape == (batch_size, seq_q, self.n_head, self.dh)
        context_vector = tf.reshape(
            context_vector, shape=(batch_size, seq_q, self.d_model)
        )

        # Project linearly
        context_vector = self.Wo(context_vector)
        assert context_vector.shape == (batch_size, seq_q, self.d_model)

        return context_vector, attention_weights


class EncoderLayer(Layer):
    def __init__(self, d_model, n_head, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(n_head, d_model)
        self.position_wise_feed_forward = PositionWiseFeedFoward(d_ff, d_model)
        self.layer_norm1 = LayerNorm()
        self.layer_norm2 = LayerNorm()

        # For example, setting rate=0.1 would drop 10% of input elements.
        self.dropout = Dropout(rate=dropout_rate)

    def call(self, x, mask, training):
        """

        Args:
             x: input tensor with shape [batch_size, seq_len, d_model]

        Returns:
            sub_layer_output: output tensor with same shape of input tensor

        """

        sub_layer_input = x
        # TODO Process attention weights
        sub_layer_output, _ = self.multi_head_attention(
            sub_layer_input, sub_layer_input, sub_layer_input, mask
        )
        # Apply dropout to the output of each sub-layer,
        # before adding sub-layer's input and normalizing
        sub_layer_output = self.dropout(sub_layer_output, trainig=training)
        sub_layer_output = sub_layer_input + sub_layer_output
        sub_layer_output = self.layer_norm1(sub_layer_output, training=training)

        # Apply FeedFowardNetwork
        sub_layer_input = sub_layer_output
        sub_layer_output = self.position_wise_feed_forward(
            sub_layer_input, training=training, padding=mask
        )
        # Apply dropout to the output of each sub-layer,
        # before adding sub-layer's input and normalizing
        sub_layer_output = self.dropout(sub_layer_output, trainig=training)
        sub_layer_output = sub_layer_input + sub_layer_output
        sub_layer_output = self.layer_norm2(sub_layer_output, training=training)

        return sub_layer_output


class Encoder(Model):
    def __init__(self, d_model, n_layer, n_head, d_ff, dropout_rate):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer

        self.encoder_layers = [
            EncoderLayer(d_model, n_head, d_ff, dropout_rate) for _ in range(n_layer)
        ]
        self.dropout = Dropout(rate=dropout_rate)

    def call(self, x, training, pad_mask, **kwargs):
        """

        Args:
            x: input tensor with shape [batch_size, seq_len, d_model]

        Returns:
            x: output tensor with shape [batch_size, seq_len, d_model]

        """
        # Apply dropout to embedded input
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask=pad_mask, training=training)

        return x


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


class SharedEmbedding(Layer):
    def __init__(self, vocab_size, d_model):
        """

        Args:
            vocab_size:
            d_model:
        """
        super(Embedding, self).__init__()

        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding(MAX_POSITION, d_model)

    def call(self, x):
        """
        Use learned embeddings to convert the input
        tokens and output tokens to vectors of dimension d_model.

        Args:
            x: input tensor with shape [batch_size, length] and dtype int32

        Returns:
            y: embedded tensor with shape [batch_size, length, d_model]
        """
        length = x.shape[-1]
        d_model = tf.cast(x.shape[-1], dtype=tf.float32)

        x = self.embedding(x)
        # Scale embedding by sqrt of the hidden size.
        x *= tf.sqrt(d_model)
        # Inject some information about the relative or absolute position of the tokens.
        x += self.positional_encoding[:, :length, :]
        return x

    def predict(self, decoder_output):
        """
        Use the usual learned linear transformation and softmax function
        to convert the decoder output to predicted next-token probabilities

        Args:
            # decoder_output: with shape [batch_size, seq_len,

        Returns:

        """
        # Share the same weight matrix between the two embedding layers and the pre-softmax
        decoder_output = tf.matmul(
            decoder_output, self.embedding.trainable_weights[0], transpose_b=True
        )
        return tf.keras.activations.softmax(decoder_output)


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
