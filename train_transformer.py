from collections import namedtuple
from time import time
import tensorflow as tf
import argparse

from src.data.data_loader import DataLoader

from src.config import Config
from src.model.loss import transformer_train_loss
from src.model.transformer import Transformer
from src.model.transformer import create_look_ahead_mask, create_pad_mask
from src.utils import get_bleu_score

pad_idx = 0
MAX_INFERENCE_LEN = 20
keras = tf.keras
PREFETCH = tf.data.experimental.AUTOTUNE
sparse_categorical_crossentropy = keras.losses.sparse_categorical_crossentropy
# Assign variables & logger
Adam = keras.optimizers.Adam
logger = tf.get_logger()

# Arguments
logger.info('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--config_json', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
args = parser.parse_args()

data_path = args.data_path
config_json = args.config_json

# Configuration
config = Config.from_json_file(config_json)
logger.setLevel(config.log_level)

# Dataloader & Dataset
data_loader = DataLoader(data_path, **config.data_loader['args'])
dataset_train = tf.data.Dataset.from_generator(
    data_loader.train_data_generator,
    output_types=(tf.int32, tf.int32)).shuffle(config.buffer_size).batch(
        config.batch_size, drop_remainder=True).prefetch(PREFETCH)
dataset_test = tf.data.Dataset.from_generator(
    data_loader.test_data_generator,
    output_types=(tf.int32, tf.int32)).shuffle(config.buffer_size).batch(
        config.inference_size, drop_remainder=True).repeat().prefetch(PREFETCH)
dataset_test_iterator = iter(dataset_test)

# Tokenizer
logger.info('Getting Tokenizer')
tokenizers = data_loader.tokenizer
tokenizer_ko: tf.keras.preprocessing.text.Tokenizer = tokenizers.kor
tokenizer_en: tf.keras.preprocessing.text.Tokenizer = tokenizers.eng
vocab_size_ko = len(tokenizer_ko.word_index) + 1
vocab_size_en = len(tokenizer_en.word_index) + 1

# Model
# encoder = Encoder(vocab_size_en, **config.encoder['args'])
# decoder = Decoder(vocab_size_ko, **config.decoder['args'])
transformer = Transformer(target_vocab_size=vocab_size_ko,
                          input_vocab_size=vocab_size_en,
                          **config.transformer['args'])

# Tensorboard
log_writer = namedtuple('logWriter', ['train', 'test'])
log_writer.train = tf.summary.create_file_writer(logdir=config.logdir +
                                                 '_train')
log_writer.test = tf.summary.create_file_writer(logdir=config.logdir + '_test')

# Optimizer
optimizer = Adam(**config.optimizer['args'])

# Checkpoint & Manager
Checkpoint = tf.train.Checkpoint
CheckpointManager = tf.train.CheckpointManager
ckpt = Checkpoint(step=tf.Variable(initial_value=0, dtype=tf.int64),
                  optimizer=optimizer,
                  transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=config.ckpt_dir,
                                          max_to_keep=config.ckpt_max_keep)

latest_checkpoint = ckpt_manager.latest_checkpoint
if latest_checkpoint:
  ckpt.restore(latest_checkpoint)
  logger.info(f'Restore from {latest_checkpoint}')
else:
  logger.info('Train from scratch')


# Training & Inference
# @tf.function
def train_step(en_train, ko_train):
  """

  Args:
    en_train: `(batch_size, seq_len_en)`
    ko_train: `(batch_size, seq_len_ko)`

  Returns:
    loss: scalar
  """
  decoder_inputs = ko_train[:, :-1]
  labels = ko_train[:, 1:]

  with tf.GradientTape() as tape:
    enc_pad_mask = create_pad_mask(en_train, pad_idx)
    dec_pad_mask = create_pad_mask(en_train, pad_idx)
    look_ahead_mask = create_look_ahead_mask(decoder_inputs.shape[1])
    self_attention_mask = create_pad_mask(decoder_inputs, pad_idx)
    look_ahead_mask = tf.maximum(self_attention_mask, look_ahead_mask)
    logits = transformer(inputs=(en_train, decoder_inputs),
                         training=True,
                         enc_pad_mask=enc_pad_mask,
                         dec_pad_mask=dec_pad_mask,
                         look_ahead_mask=look_ahead_mask)

    loss = transformer_train_loss(logits, labels, pad_idx)

  trainable_variables = transformer.trainable_variables
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss


# @tf.function
def inference(data):
  batch_size = data.shape[0]

  input_decoder = tf.expand_dims([tokenizer_en.word_index['<start>']] *
                                 batch_size, 1)

  for i in range(MAX_INFERENCE_LEN):
    enc_pad_mask = create_pad_mask(data, pad_idx)
    dec_self_attention_mask = create_pad_mask(input_decoder, pad_idx)
    look_ahead_mask = create_look_ahead_mask(input_decoder.shape[1])
    look_ahead_mask = tf.maximum(dec_self_attention_mask, look_ahead_mask)
    dec_pad_mask = create_pad_mask(data, pad_idx)

    logits = transformer(inputs=(data, input_decoder),
                         training=False,
                         enc_pad_mask=enc_pad_mask,
                         dec_pad_mask=dec_pad_mask,
                         look_ahead_mask=look_ahead_mask)
    logits = logits[:, -1, :]
    probs = tf.math.softmax(logits, axis=-1)
    preds = tf.argmax(probs, axis=-1)
    preds = tf.expand_dims(preds, 1)
    preds = tf.cast(preds, tf.int32)
    input_decoder = tf.concat([input_decoder, preds], axis=-1)
  return input_decoder


for e in range(config.epochs):
  logger.info(f'Train epoch: {e}')
  for en, ko in dataset_train:
    step = ckpt.step.numpy()
    start = time()
    train_loss = train_step(en, ko)
    end = time()
    train_step_time = end - start

    with log_writer.train.as_default():
      tf.summary.scalar('train_loss', train_loss, step)
      tf.summary.scalar('train_step_time', train_step_time, step)

    if step % config.display_step == 0:
      logger.info(f'Train step: {step}')
      logger.info(f'  Loss: {train_loss}')
      logger.info(f'  Time: {train_step_time : 0.2f}')

      logger.info('Train Inferences')
      original_eng_text = tokenizer_en.sequences_to_texts(en.numpy()[:2])
      logger.info(f'  original eng text: {original_eng_text}')
      ko_inferenced = inference(en)
      original_kor_text = tokenizer_ko.sequences_to_texts(ko.numpy()[:2])
      logger.info(f'  original kor text: {original_kor_text}')
      inferenced_kor_text = tokenizer_ko.sequences_to_texts(
          ko_inferenced.numpy()[:2])
      logger.info(f'  inferenced kor text: {inferenced_kor_text}')
      ko = tokenizer_ko.sequences_to_texts(ko.numpy())
      ko_inferenced = tokenizer_ko.sequences_to_texts(ko_inferenced.numpy())
      bleu_train = get_bleu_score(ko, ko_inferenced)
      logger.info(f'  mean bleu score: {bleu_train}')

      with log_writer.train.as_default():
        tf.summary.text('original_eng_text', original_eng_text, step)
        tf.summary.text('original_kor_text', original_kor_text, step)
        tf.summary.text('inferenced_kor_text', inferenced_kor_text, step)
        tf.summary.scalar('bleu', bleu_train, step)

      logger.info(f'Test Inferences')
      en, ko = next(dataset_test_iterator)
      original_eng_text = tokenizer_en.sequences_to_texts(en.numpy())
      logger.info(f'  original eng text: {original_eng_text}')
      ko_inferenced = inference(en)
      original_kor_text = tokenizer_ko.sequences_to_texts(ko.numpy())
      logger.info(f'  original kor text: {original_kor_text}')
      inferenced_kor_text = tokenizer_ko.sequences_to_texts(
          ko_inferenced.numpy())
      logger.info(f'  inferenced kor text: {inferenced_kor_text}')
      ko = tokenizer_ko.sequences_to_texts(ko.numpy())
      ko_inferenced = tokenizer_ko.sequences_to_texts(ko_inferenced.numpy())
      bleu_test = get_bleu_score(ko, ko_inferenced)
      logger.info(f'  mean bleu score: {bleu_test}')

      with log_writer.test.as_default():
        tf.summary.text('original_eng_text', original_eng_text, step)
        tf.summary.text('original_kor_text', original_kor_text, step)
        tf.summary.text('inferenced_kor_text', inferenced_kor_text, step)
        tf.summary.scalar('bleu', bleu_test, step)

    if step % config.save_step == 0:
      logger.info(f'Save model at step {step}')
      ckpt_manager.save()

    ckpt.step.assign_add(1)