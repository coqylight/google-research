# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Trains a single model replica against a single dataset.

For an ensemble as used within the manuscript, N of these model replicas are
trained separately with distinct randomized weight initializations.


Run CNN:
train.py \
    --model=cnn \
    --seq_encoder=fixedlen-id \
    --model_dir=/tmp/cnn_model_dir \
    --train_path=/path/to/datasets/train.tfrecord \
    --validation_path=/path/to/datasets/valid.tfrecord \
    --alsologtostderr

Run RNN:
train.py \
    --model=rnn \
    --seq_encoder=varlen-id \
    --model_dir=/tmp/rnn_model_dir \
    --train_path=/path/to/datasets/train.tfrecord \
    --validation_path=/path/to/datasets/valid.tfrecord \
    --alsologtostderr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import flags
from absl import logging
import cnn_torch
import lr
import rnn
import tensorflow as tf
from tensorflow import estimator as tf_estimator
import train_utils
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ..util import dataset_utils


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'model_dir', None, 'Path to dir for training checkpoints.')
flags.DEFINE_string(
    'train_path',
    '/namespace/gas/primary/wrangler/models/datasets/train.tfrecord',
    'Path to train dataset.')
flags.DEFINE_string(
    'validation_path',
    '/namespace/gas/primary/wrangler/models/datasets/valid.tfrecord',
    'Path to validation dataset.')
flags.DEFINE_enum(
    'model',
    None,
    ['rnn', 'cnn', 'logistic'],
    'Model architecture.')
flags.DEFINE_enum(
    'seq_encoder',
    None,
    ['varlen-id', 'fixedlen-id'],
    'Sequence encoder.')
flags.DEFINE_string(
    'hparams',
    None,
    'Model hyperparameter overrides: "hparam1=value1,hparam2=value2".')
flags.DEFINE_enum(
    'eval_metric',
    'precision',
    ['precision', 'recall', 'accuracy'],
    'Evaluation set metric to monitor for early stopping.')
flags.DEFINE_integer(
    'train_steps_per_eval',
    500,
    'Train steps per evaluation; should be >= # steps per training epoch.')
flags.DEFINE_integer(
    'max_train_steps',
    100000,
    'Maximum number of training steps to run.')
flags.DEFINE_integer(
    'early_stopper_num_evals_to_wait',
    10,
    'Number of evaluations to wait for eval_metric to improve before stopping.')
flags.DEFINE_string(
    'ref_seq',
    'DEEEIRTTNPVATEQYGSVSTNLQRGNR',
    'Reference sequence for mutation-based encodings.')
flags.DEFINE_integer(
    'eval_throttle_secs',
    10,
    'Minimum number of seconds to wait, from training start or last eval.')

_DEFAULT_HPARAMS_RNN = {
    'model': 'rnn',
    'seq_encoder': 'varlen-id',
    'batch_size': 25,
    'learning_rate': .01,
    'num_classes': 2,
    'num_layers': 2,
    'num_units': 100,
}

_DEFAULT_HPARAMS_CNN = {
    'model': 'cnn',
    'seq_encoder': 'fixedlen-id',
    'batch_size': 25,
    'learning_rate': 0.001,
    'num_classes': 2,
    'residue_encoding_size': None,
    'seq_encoding_length': None,
    'pool_width': 2,
    'conv_depth': 12,
    'conv_depth_multiplier': 2,
    'conv_width': 7,
    'feature_axis': -1,
    'fc_size': 128,
    'fc_size_multiplier': 0.5,
}

_DEFAULT_HPARAMS_LOGISTIC = {
    'model': 'logistic',
    'seq_encoder': 'fixedlen-id',
    'batch_size': 25,
    'learning_rate': 0.01,
    'num_classes': 2,
}


class _MetricAccumulator(object):
  """Accumulates metrics weighted by batch size."""

  def __init__(self):
    self._totals = collections.defaultdict(float)
    self._count = 0
    self._updated_in_step = False

  def update(self, metrics):
    batch_size = int(metrics.get('batch_size', 1))
    self._count += batch_size
    for key, value in metrics.items():
      if key == 'batch_size':
        continue
      self._totals[key] += float(value) * batch_size
    self._updated_in_step = True

  def consume_pending_update(self):
    was_updated = self._updated_in_step
    self._updated_in_step = False
    return was_updated

  def compute(self):
    if not self._totals:
      return {}
    if self._count <= 0:
      return {key: 0.0 for key in self._totals}
    return {key: total / self._count for key, total in self._totals.items()}

  def reset(self):
    self._totals = collections.defaultdict(float)
    self._count = 0
    self._updated_in_step = False


def _load_dataset_tensors(file_path, seq_encoder_fn, batch_size):
  """Loads a TFRecord dataset into NumPy arrays."""
  sequences = []
  labels = []
  with tf.Graph().as_default():
    dataset = dataset_utils.read_tfrecord_dataset([file_path]).map(seq_encoder_fn)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_features, next_labels = iterator.get_next()
    with tf.Session() as sess:
      while True:
        try:
          features_batch, labels_batch = sess.run([next_features, next_labels])
        except tf.errors.OutOfRangeError:
          break
        sequences.append(features_batch['sequence'])
        labels.append(labels_batch)

  if not sequences:
    raise ValueError('No examples found in dataset: %s' % file_path)

  features_np = numpy.concatenate(sequences, axis=0).astype(numpy.float32, copy=False)
  labels_np = numpy.concatenate(labels, axis=0).astype(numpy.int64, copy=False)
  return features_np, labels_np


def _build_dataloader(features, labels, batch_size, shuffle, drop_last):
  dataset = TensorDataset(
      torch.from_numpy(features),
      torch.from_numpy(labels))
  return DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last)


def _evaluate_model(model, data_loader, device, positive_class):
  if data_loader is None:
    return {}

  model.eval()
  accumulator = _MetricAccumulator()
  loss_fn = nn.CrossEntropyLoss()
  with torch.no_grad():
    for features, labels in data_loader:
      if device is not None:
        features = features.to(device)
        labels = labels.to(device)
      logits = model(features)
      loss = loss_fn(logits, labels)
      metrics = cnn_torch.compute_classification_metrics(
          logits,
          labels,
          positive_class)
      metrics['loss'] = loss.item()
      metrics['batch_size'] = int(labels.size(0))
      accumulator.update(metrics)
      accumulator.consume_pending_update()
  model.train()
  return accumulator.compute()


def _generic_pytorch_training_loop(
    model,
    train_loader,
    validation_loader,
    step_fn,
    train_accumulator,
    max_steps,
    eval_interval,
    device,
    positive_class):
  if len(train_loader) == 0:
    raise ValueError('Training dataset is empty.')

  history = []
  eval_interval = max(1, eval_interval)
  steps = 0
  train_accumulator.reset()
  train_iterator = iter(train_loader)

  while steps < max_steps:
    try:
      batch = next(train_iterator)
    except StopIteration:
      train_iterator = iter(train_loader)
      batch = next(train_iterator)

    metrics = step_fn(batch)
    if not train_accumulator.consume_pending_update():
      train_accumulator.update(metrics)
      train_accumulator.consume_pending_update()

    steps += 1
    if (steps % eval_interval) == 0 or steps >= max_steps:
      train_metrics = train_accumulator.compute()
      validation_metrics = _evaluate_model(
          model,
          validation_loader,
          device,
          positive_class)
      history.append({
          'step': steps,
          'train': train_metrics,
          'validation': validation_metrics,
      })
      logging.info('Step %d training metrics: %s', steps, train_metrics)
      logging.info('Step %d validation metrics: %s', steps, validation_metrics)
      train_accumulator.reset()

  return history


def _train_cnn_with_pytorch(seq_encoder_fn, hparams):
  params = dict(hparams.values())
  batch_size = params.get('batch_size', _DEFAULT_HPARAMS_CNN['batch_size'])
  learning_rate = params.get('learning_rate', _DEFAULT_HPARAMS_CNN['learning_rate'])

  train_features, train_labels = _load_dataset_tensors(
      FLAGS.train_path,
      seq_encoder_fn,
      batch_size)
  validation_features, validation_labels = _load_dataset_tensors(
      FLAGS.validation_path,
      seq_encoder_fn,
      batch_size)

  train_loader = _build_dataloader(
      train_features,
      train_labels,
      batch_size=batch_size,
      shuffle=True,
      drop_last=False)
  validation_loader = _build_dataloader(
      validation_features,
      validation_labels,
      batch_size=batch_size,
      shuffle=False,
      drop_last=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Training CNN model on device: %s', device)
  model = cnn_torch.CnnClassifier(**params).to(device)

  train_accumulator = _MetricAccumulator()
  step_fn, optimizer = cnn_torch.make_training_step(
      model,
      learning_rate=learning_rate,
      device=device,
      positive_class=params.get('positive_class', model.positive_class),
      metric_hooks=[train_accumulator.update])

  history = _generic_pytorch_training_loop(
      model,
      train_loader,
      validation_loader,
      step_fn,
      train_accumulator,
      max_steps=FLAGS.max_train_steps,
      eval_interval=FLAGS.train_steps_per_eval,
      device=device,
      positive_class=params.get('positive_class', model.positive_class))

  return model, optimizer, history


def train_model(
    model_fn,
    train_input_fn,
    validation_input_fn,
    params):
  """Trains a model.

  Args:
    model_fn: (fn) A tf.Estimator model_fn.
    train_input_fn: (fn) A tf.Estimator input_fn for the training data.
    validation_input_fn: (fn) A tf.Estimator input_fn for the validation data.
    params: (dict) Model hyperparameters.
  """
  run_config = tf_estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.train_steps_per_eval,
      keep_checkpoint_max=None)

  logging.warn('RUN CONFIG: %r', run_config)

  model = tf_estimator.Estimator(
      model_fn=model_fn,
      params=params,
      config=run_config)

  experiment = tf.contrib.learn.Experiment(
      model,
      train_input_fn=train_input_fn,
      eval_input_fn=validation_input_fn,
      train_steps=FLAGS.max_train_steps,
      eval_steps=None,
      eval_delay_secs=FLAGS.eval_throttle_secs,
      train_steps_per_iteration=FLAGS.train_steps_per_eval)

  # WARNING: train_steps_per_iteration should be >= train epoch size, because
  # the train input queue is reset upon each evaluation in the Experiment
  # implementation currently; i.e., you might only ever train on a subset of the
  # training data if you configure train_steps_per_iteration < epoch size.
  #
  # See https://github.com/tensorflow/tensorflow/issues/11013
  precision_early_stopper = train_utils.EarlyStopper(
      num_evals_to_wait=FLAGS.early_stopper_num_evals_to_wait,
      metric_key=FLAGS.eval_metric)
  experiment.continuous_train_and_eval(
      continuous_eval_predicate_fn=(
          precision_early_stopper.early_stop_predicate_fn))


def main(_):
  seq_encoder = None
  seq_encoder_fn = None
  if FLAGS.seq_encoder == 'varlen-id':
    seq_encoder = dataset_utils.ONEHOT_VARLEN_SEQUENCE_ENCODER
    seq_encoder_fn = lambda ex: dataset_utils.encode_varlen(ex, seq_encoder)
  elif FLAGS.seq_encoder == 'fixedlen-id':
    seq_encoder = dataset_utils.ONEHOT_FIXEDLEN_MUTATION_ENCODER
    seq_encoder_fn = lambda ex: dataset_utils.encode_fixedlen(ex, seq_encoder)
  else:
    raise NotImplementedError(
        'Sequence encoder "%s" is not supported.' % FLAGS.seq_encoder)

  # Note that the encoding length used below provides 1 slot for each WT
  # position plus a prefix position; all of these positions can also have a
  # single insertion, making the max number of residues representable
  # 58 = 2 * (28 + 1), given the 28 WT AAV2 residue positions:
  # DEEEIRTTNPVATEQYGSVSTNLQRGNR
  default_hparams = None
  model_fn = None
  use_pytorch = False
  if FLAGS.model == 'cnn':
    default_hparams = dict(_DEFAULT_HPARAMS_CNN)
    default_hparams['residue_encoding_size'] = seq_encoder.encoding_size
    default_hparams['seq_encoding_length'] = (len(FLAGS.ref_seq) + 1) * 2
    use_pytorch = True
  elif FLAGS.model == 'rnn':
    default_hparams = _DEFAULT_HPARAMS_RNN
    model_fn = rnn.rnn_model_fn
  elif FLAGS.model == 'logistic':
    default_hparams = dict(_DEFAULT_HPARAMS_LOGISTIC)
    model_fn = lr.logistic_regression_model_fn
    default_hparams['residue_encoding_size'] = seq_encoder.encoding_size
    default_hparams['seq_encoding_length'] = (len(FLAGS.ref_seq) + 1) * 2
  else:
    raise NotImplementedError('Model type "%s" is not supported.' % FLAGS.model)

  hparams = tf.contrib.training.HParams(**default_hparams)
  if FLAGS.hparams:
    logging.info('Overriding hyperparameters with %r', FLAGS.hparams)
    hparams.parse(FLAGS.hparams)
    # Add FLAGS-based hparams that are required.
  hparams.add_param('model', FLAGS.model)
  hparams.add_param('seq_encoder', FLAGS.seq_encoder)

  os.makedirs(FLAGS.model_dir, exist_ok=True)

  if use_pytorch:
    _train_cnn_with_pytorch(seq_encoder_fn, hparams)
  else:
    train_input_fn = dataset_utils.as_estimator_input_fn(
        (dataset_utils.read_tfrecord_dataset(FLAGS.train_path)
         .map(seq_encoder_fn)),
        batch_size=hparams.batch_size,
        sequence_element_encoding_shape=(
            seq_encoder.encoding_size if FLAGS.model == 'rnn' else None),
        drop_partial_batches=True,
        num_epochs=None,
        shuffle=True)
    validation_input_fn = dataset_utils.as_estimator_input_fn(
        (dataset_utils.read_tfrecord_dataset(FLAGS.validation_path)
         .map(seq_encoder_fn)),
        batch_size=hparams.batch_size,
        sequence_element_encoding_shape=(
            seq_encoder.encoding_size if FLAGS.model == 'rnn' else None),
        drop_partial_batches=True,
        num_epochs=1,
        shuffle=False)

    train_model(
        model_fn,
        train_input_fn,
        validation_input_fn,
        hparams)

  with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
    f.write(hparams.to_json())


if __name__ == '__main__':
  flags.mark_flag_as_required('model')
  flags.mark_flag_as_required('seq_encoder')
  flags.mark_flag_as_required('model_dir')
  tf.app.run(main)
