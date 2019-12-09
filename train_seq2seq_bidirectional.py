"""
Neural machine translation english to korean
"""
import argparse
from collections import namedtuple
from time import time
import tensorflow as tf

from src.data.data_loader import DataLoader
from src.config import Config
from src.utils import get_bleu_score
from src.model.seq2seq_bidirectional import Encoder, Decoder

PREFETCH = tf.data.experimental.AUTOTUNE

keras = tf.keras
Adam = keras.optimizers.Adam
logger = tf.get_logger()

# Argument
logger.info('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--config_json', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--deu', type=bool, required=False, default=False)
args = parser.parse_args()

data_path = args.data_path
config_json = args.config_json
deu = args.deu

# Config
config = Config.from_json_file(config_json)
logger.setLevel(config.log_level)

# DataLoader
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
tokenizer_ori: tf.keras.preprocessing.text.Tokenizer = tokenizers.ori
tokenizer_tar: tf.keras.preprocessing.text.Tokenizer = tokenizers.tar
vocab_size_ko = data_loader.ori_vocab_size
vocab_size_en = data_loader.tar_vocab_size

# Model
encoder = Encoder(vocab_size_en, **config.encoder['args'])
decoder = Decoder(vocab_size_ko, **config.decoder['args'])

# Optimizer
optimizer = Adam(**config.optimizer['args'])


@tf.function
def train_step(ori_train, tar_train):
  loss = 0
  train_batch_size = config.batch_size
  mask = tf.zeros_like(tar_train, dtype=tf.bool)
  mask |= tf.cast(tar_train, tf.bool)

  with tf.GradientTape() as tape:
    initial_state = encoder.initial_state(train_batch_size)
    inputs_encoder = (ori_train, initial_state)
    outputs_encoder, h_f_encoder, h_b_encoder, c_f_encoder, c_b_encoder = encoder(
        inputs_encoder)

    # hidden representation of ori_tarin
    h_decoder = tf.concat([h_f_encoder, h_b_encoder], axis=-1)
    c_decoder = tf.concat([c_f_encoder, c_b_encoder], axis=-1)

    # decoder's input 0 start from <start> token
    input_decoder = tf.expand_dims([tokenizer_ori.word_index['<start>']] *
                                   train_batch_size, 1)    # [batch_size, 1]

    len_tar = range(1, tar_train.shape[1])
    for t in len_tar:
      initial_state = (h_decoder, c_decoder)
      inputs_decoder = (input_decoder, initial_state)
      logits, h_decoder, c_decoder = decoder(inputs_decoder)

      # logits - decoder's prediction at timestep t-1,
      # they are corresponding to encoder's input at timestep t
      # total loss is summation of every timestep's losses
      labels = tf.reshape(tar_train[:, t], shape=(-1, 1))
      time_loss = keras.losses.sparse_categorical_crossentropy(y_true=labels,
                                                               y_pred=logits,
                                                               from_logits=True)
      time_loss *= tf.cast(tf.reshape(mask[:, t], (-1,)),
                           dtype=tf.float32)
      loss += time_loss
      # Teacher forcing - at training time, only labels are fed
      input_decoder = tf.expand_dims(tar_train[:, t],
                                     1)    # [batch_size, 1]
    loss = tf.reduce_mean(loss)

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss


@tf.function
def inference(inference_data):
  inference_size = inference_data.shape[0]
  initial_state = encoder.initial_state(inference_size)
  inputs_inference = (inference_data, initial_state)

  outputs_encoder, h_f_encoder, h_b_encoder, c_f_encoder, c_b_encoder = encoder(
      inputs_inference)

  h_decoder = tf.concat([h_f_encoder, h_b_encoder], axis=-1)
  c_decoder = tf.concat([c_f_encoder, c_b_encoder], axis=-1)

  input_decoder = tf.expand_dims([tokenizer_ori.word_index['<start>']] *
                                 inference_size,
                                 axis=1)

  max_timestep = outputs_encoder.shape[1]

  ret = tf.zeros((inference_size, 0), dtype=tf.int32)

  for timestep in range(max_timestep):
    inputs_decoder = input_decoder, (h_decoder, c_decoder)
    outputs_decoder = decoder(inputs_decoder, training=False)
    logits, h_decoder, c_decoder = outputs_decoder

    # [time_steps, 1]
    probs = tf.math.softmax(logits, axis=-1)
    preds = tf.argmax(probs, axis=-1)
    preds = tf.expand_dims(preds, 1)
    preds = tf.cast(preds, dtype=tf.int32)
    ret = tf.concat([ret, preds], axis=-1)
    input_decoder = preds
  return ret


# Tensorboard
log_writer = namedtuple('logWriter', ['train', 'test'])
log_writer.train = tf.summary.create_file_writer(logdir=config.logdir +
                                                 '_train')
log_writer.test = tf.summary.create_file_writer(logdir=config.logdir + '_test')

# Checkpoint & Manager
Checkpoint = tf.train.Checkpoint
ckpt = Checkpoint(step=tf.Variable(initial_value=0, dtype=tf.int64),
                  optimizer=optimizer,
                  encoder=encoder,
                  decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=config.ckpt_dir,
                                          max_to_keep=config.ckpt_max_keep)

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
    train_step_time = end - start

    with log_writer.train.as_default():
      tf.summary.scalar('train_loss', train_loss, step)
      tf.summary.scalar('train_step_time', train_step_time, step)

    if step % config.display_step == 0:
      logger.info(f'Train step: {step}')
      logger.info(f'  Loss: {train_loss}')
      logger.info(f'  Time: {train_step_time : 0.2f}')

      logger.info('Train Inferences')
      original_eng_text = tokenizer_ori.sequences_to_texts(en.numpy()[:2])
      logger.info(f'  original eng text: {original_eng_text}')
      ko_inferenced = inference(en)
      original_kor_text = tokenizer_tar.sequences_to_texts(ko.numpy()[:2])
      logger.info(f'  original kor text: {original_kor_text}')
      inferenced_kor_text = tokenizer_tar.sequences_to_texts(
          ko_inferenced.numpy()[:2])
      logger.info(f'  inferenced kor text: {inferenced_kor_text}')
      ko = tokenizer_tar.sequences_to_texts(ko.numpy())
      ko_inferenced = tokenizer_tar.sequences_to_texts(ko_inferenced.numpy())
      bleu_train = get_bleu_score(ko, ko_inferenced)
      logger.info(f'  mean bleu score: {bleu_train}')

      with log_writer.train.as_default():
        tf.summary.text('original_eng_text', original_eng_text, step)
        tf.summary.text('original_kor_text', original_kor_text, step)
        tf.summary.text('inferenced_kor_text', inferenced_kor_text, step)
        tf.summary.scalar('bleu', bleu_train, step)

      logger.info(f'Test Inferences')
      en, ko = next(dataset_test_iterator)
      original_eng_text = tokenizer_ori.sequences_to_texts(en.numpy())
      logger.info(f'  original eng text: {original_eng_text}')
      ko_inferenced = inference(en)
      original_kor_text = tokenizer_tar.sequences_to_texts(ko.numpy())
      logger.info(f'  original kor text: {original_kor_text}')
      inferenced_kor_text = tokenizer_tar.sequences_to_texts(
          ko_inferenced.numpy())
      logger.info(f'  inferenced kor text: {inferenced_kor_text}')
      ko = tokenizer_tar.sequences_to_texts(ko.numpy())
      ko_inferenced = tokenizer_tar.sequences_to_texts(ko_inferenced.numpy())
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
