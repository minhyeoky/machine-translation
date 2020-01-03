import tensorflow as tf

keras = tf.keras
sparse_categorical_crossentropy = keras.losses.sparse_categorical_crossentropy


def transformer_train_loss(logits, labels, pad_idx):
    mask = tf.cast(tf.not_equal(labels, pad_idx), dtype=tf.float32)
    loss = sparse_categorical_crossentropy(labels, logits, from_logits=True, axis=-1)
    loss = loss * mask
    loss = tf.reduce_mean(loss)
    return loss
