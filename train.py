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
logger = tf.get_logger()
logger.setLevel('DEBUG')

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
data_loader = DataLoader(data_path, n_data=config.n_data, test_size=config.test_size)
dataset_train = tf.data.Dataset.from_generator(data_loader.train_data_generator,
                                               output_types=(tf.int32, tf.int32)).shuffle(
    config.buffer_size).batch(config.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
dataset_test = tf.data.Dataset.from_generator(data_loader.test_data_generator,
                                              output_types=(tf.int32, tf.int32)).shuffle(
    config.buffer_size).batch(config.inference_size, drop_remainder=True).repeat().prefetch(
    tf.data.experimental.AUTOTUNE)
dataset_test_iterator = iter(dataset_test)

# Tokenizer
logger.info('Getting Tokenizer')
tokenizers = data_loader.tokenizer
tokenizer_ko: tf.keras.preprocessing.text.Tokenizer = tokenizers.kor
tokenizer_en: tf.keras.preprocessing.text.Tokenizer = tokenizers.eng
vocab_size_ko = len(tokenizer_ko.word_index) + 1
vocab_size_en = len(tokenizer_en.word_index) + 1

# Model
encoder = Encoder(vocab_size_en, embedding_size=config.embedding_size, n_units=config.n_units)
decoder = Decoder(vocab_size_ko, embedding_size=config.embedding_size, n_units=config.n_units)

# Optimizer
optimizer = Adam(config.lr, config.beta_1)


# @tf.function
def train_step(en_train, ko_train):
    loss = 0
    mask = tf.zeros_like(ko_train, dtype=tf.bool)
    mask |= tf.cast(ko_train, tf.bool)

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
            time_loss = keras.losses.sparse_categorical_crossentropy(y_true=labels,
                                                                     y_pred=logits,
                                                                     from_logits=True)
            time_loss *= tf.cast(tf.reshape(mask[:, time_step], (-1,)), dtype=tf.float32)
            loss += time_loss
            # Teacher forcing - at training time, only labels are fed
            input_decoder = tf.expand_dims(ko_train[:, time_step], 1)  # [batch_size, 1]
        loss = tf.reduce_mean(loss)

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss


# @tf.function
def inference(inference_data):
    inference_size = inference_data.shape[0]
    initial_state = encoder.initial_state(inference_size)
    inputs = inference_data, initial_state

    outputs, h_encoder, c_encoder = encoder(inputs, training=False)

    h_decoder, c_decoder = h_encoder, c_encoder

    input_decoder = tf.expand_dims([tokenizer_en.word_index['<start>']] * inference_size, axis=1)

    max_timestep = outputs.shape[1]

    test_inferenced = tf.zeros((inference_size, 0), dtype=tf.int32)

    for timestep in range(max_timestep):
        inputs_decoder = input_decoder, (h_decoder, c_decoder)
        outputs_decoder = decoder(inputs_decoder, training=False)
        logits, h_decoder, c_decoder = outputs_decoder

        # [time_steps, 1]
        probs = tf.math.softmax(logits, axis=-1)
        preds = tf.argmax(probs, axis=-1)
        preds = tf.expand_dims(preds, 1)
        preds = tf.cast(preds, dtype=tf.int32)
        test_inferenced = tf.concat([test_inferenced, preds], axis=-1)
        input_decoder = preds

        # for b in range(inference_size):
        #     test_inferenced[b].append(preds[b])
        #     ko_index_word_preds_b_ = tokenizer_ko.index_word[preds[b]]
        #     if ko_index_word_preds_b_ == tokenizer_ko.oov_token:
        #         continue
        #     else:
        #         test_inferenced[b].append(ko_index_word_preds_b_)
    return test_inferenced


# Tensorboard
log_writer = tf.summary.create_file_writer(logdir=config.logdir)

# Checkpoint & Manager
Checkpoint = tf.train.Checkpoint
ckpt = Checkpoint(step=tf.Variable(initial_value=0, dtype=tf.int64), optimizer=optimizer,
                  encoder=encoder, decoder=decoder)

ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=config.save_dir, max_to_keep=10)

# Load checkpoint if exists
latest_checkpoint = ckpt_manager.latest_checkpoint
if latest_checkpoint:
    ckpt.restore(latest_checkpoint)
    logger.info(f'Restore from {latest_checkpoint}')
else:
    logger.info('Train from scratch')

for epoch in range(config.epochs):
    logger.info(f'Train epoch: {epoch}')

    for en, ko in dataset_train:
        step = ckpt.step.numpy()

        start = time()
        train_loss = train_step(en, ko)
        end = time()
        if step % config.display_step == 0:
            logger.info(f'Train step: {step}')
            logger.info(f'  Loss: {train_loss}')
            logger.info(f'  Time: {end - start: 0.2f}')

            logger.info('Train Inferences')
            logger.info(f'  original eng text: {tokenizer_en.sequences_to_texts(en.numpy()[:2])}')
            ko_inferenced = inference(en)
            logger.info(f'  original kor text: {tokenizer_ko.sequences_to_texts(ko.numpy()[:2])}')
            logger.info(f'  inferenced kor text: {tokenizer_ko.sequences_to_texts(ko_inferenced.numpy()[:2])}')

            logger.info(f'Test Inferences')
            en, ko = next(dataset_test_iterator)
            logger.info(f'  original eng text: {tokenizer_en.sequences_to_texts(en.numpy())}')
            ko_inferenced = inference(en)
            logger.info(f'  original kor text: {tokenizer_ko.sequences_to_texts(ko.numpy())}')
            logger.info(f'  inferenced kor text: {tokenizer_ko.sequences_to_texts(ko_inferenced.numpy())}')
        if step % config.save_step == 0:
            logger.info(f'Save model at step {step}')
            ckpt_manager.save()

        ckpt.step.assign_add(1)
