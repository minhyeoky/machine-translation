import argparse
from time import time

import tensorflow as tf
import logging

from src.model.model import Encoder, Decoder
from src.data.data_loader import DataLoader
from src.config import Config

"""
Neural machine translation english to korean
"""
keras = tf.keras
Adam = keras.optimizers.Adam
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

# Argument
logger.info('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--config_json', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
args = parser.parse_args()

data_path = args.data_path
config_json = args.config_json

# Config
config = Config.from_json_file(config_json)

# DataLoader
data_loader = DataLoader(data_path)
dataset = tf.data.Dataset.from_generator(data_loader.train_data_generator, output_types=(tf.int32, tf.int32)).shuffle(
    config.buffer_size).batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Tokenizer
logger.info('Getting Tokenizer')
tokenizers = data_loader.tokenizer
tokenizer_ko: tf.keras.preprocessing.text.Tokenizer = tokenizers.kor
tokenizer_en: tf.keras.preprocessing.text.Tokenizer = tokenizers.eng
vocab_size_ko = len(tokenizer_ko.word_index)
vocab_size_en = len(tokenizer_en.word_index)

# Model
encoder = Encoder(vocab_size_en, embedding_size=config.embedding_size, n_units=config.n_units)
decoder = Decoder(vocab_size_ko, embedding_size=config.embedding_size, n_units=config.n_units)

# Optimizer
optimizer = Adam(config.lr, config.beta_1)


# @tf.function
def train_step(en_train, ko_train):
    loss = 0

    with tf.GradientTape() as tape:
        initial_state = encoder.initial_state(config.batch_size)
        outputs_encoder, h_encoder, c_encoder = encoder((en_train, initial_state))

        # hidden representation of en_train
        h_decoder, c_decoder = h_encoder, c_encoder

        # decoder's input 0 start from <start> token
        input_decoder = tf.expand_dims([tokenizer_en.word_index['<start>']] * config.batch_size, 1)  # [batch_size, 1]

        for time_step in range(1, outputs_encoder.shape[1]):
            initial_state = (h_decoder, c_decoder)
            logits, h_decoder, c_decoder = decoder((input_decoder, initial_state))

            # logits - decoder's prediction at timestep t-1,
            # they are corresponding to encoder's input at timestep t
            # total loss is summation of every timestep's losses
            labels = tf.reshape(ko_train[:, time_step], shape=(-1, 1))
            loss += keras.losses.sparse_categorical_crossentropy(y_true=labels,
                                                                 y_pred=logits,
                                                                 from_logits=True)
            # Teacher forcing - at training time, only labels are fed
            input_decoder = tf.expand_dims(ko_train[:, time_step], 1)  # [batch_size, 1]
        loss = tf.reduce_mean(loss)

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss


step = 0

for epoch in range(config.epochs):
    logger.info(f'Train epoch: {epoch}')

    for en, ko in dataset:
        start = time()
        train_loss = train_step(en, ko)
        end = time()
        if step % config.display_step == 0:
            logger.info(f'Train step: {step}')
            logger.info(f'  Loss: {train_loss}')
            logger.info(f'  Time: {end - start: 0.2f}')
