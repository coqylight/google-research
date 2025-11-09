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

import functools
import os
from typing import Iterable, Optional, Sequence, Tuple, Union

from absl import flags
from absl import logging
import rnn

tf = None
tf_estimator = None
cnn = None
lr = None
train_utils = None
dataset_utils = None

import torch
from torch import nn
from torch.utils import data as torch_data


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


DatasetLike = Union[torch_data.Dataset, torch_data.DataLoader, Sequence]


def _ensure_tensorflow():
  """Lazily imports TensorFlow dependencies."""
  global tf  # pylint: disable=global-variable-undefined
  global tf_estimator  # pylint: disable=global-variable-undefined
  if tf is None or tf_estimator is None:
    try:
      import tensorflow as tf_module  # pylint: disable=g-import-not-at-top
      from tensorflow import estimator as tf_estimator_module  # pylint: disable=g-import-not-at-top
    except Exception as exc:  # pylint: disable=broad-except
      raise ImportError('TensorFlow dependencies are unavailable.') from exc
    tf = tf_module
    tf_estimator = tf_estimator_module


def _ensure_cnn():
  """Lazily imports the CNN training module."""
  global cnn  # pylint: disable=global-variable-undefined
  if cnn is None:
    try:
      from . import cnn as cnn_module  # pylint: disable=g-import-not-at-top
    except Exception as exc:  # pylint: disable=broad-except
      raise ImportError('CNN model dependencies are unavailable.') from exc
    cnn = cnn_module


def _ensure_lr():
  """Lazily imports the logistic regression module."""
  global lr  # pylint: disable=global-variable-undefined
  if lr is None:
    try:
      from . import lr as lr_module  # pylint: disable=g-import-not-at-top
    except Exception as exc:  # pylint: disable=broad-except
      raise ImportError('Logistic regression model dependencies are unavailable.') from exc
    lr = lr_module


def _ensure_train_utils():
  """Lazily imports TensorFlow training utilities."""
  global train_utils  # pylint: disable=global-variable-undefined
  if train_utils is None:
    try:
      from . import train_utils as train_utils_module  # pylint: disable=g-import-not-at-top
    except Exception as exc:  # pylint: disable=broad-except
      raise ImportError('TensorFlow training utilities are unavailable.') from exc
    train_utils = train_utils_module


def _ensure_dataset_utils():
  """Lazily imports dataset utilities."""
  global dataset_utils  # pylint: disable=global-variable-undefined
  if dataset_utils is None:
    try:
      from ..util import dataset_utils as dataset_utils_module  # pylint: disable=g-import-not-at-top
    except Exception as exc:  # pylint: disable=broad-except
      raise ImportError('Dataset utilities are unavailable.') from exc
    dataset_utils = dataset_utils_module


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
  _ensure_tensorflow()
  _ensure_train_utils()

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


def _extract_example_components(example):
  """Normalizes raw dataset examples to (sequence, length, label)."""
  length = None
  if isinstance(example, dict):
    sequence = example.get('sequence')
    length = example.get('sequence_length')
    label = example.get('label')
    if label is None:
      label = example.get('is_viable')
    if label is None and 'labels' in example:
      label = example['labels']
  else:
    if isinstance(example, tuple) or isinstance(example, list):
      if len(example) == 3:
        sequence, length, label = example
      elif len(example) == 2:
        sequence, label = example
      else:
        raise ValueError('Unsupported example structure: %r' % (example,))
    else:
      raise TypeError('Unsupported example type: %s' % type(example))

  if sequence is None or label is None:
    raise ValueError('Dataset example must provide sequence and label fields.')

  sequence_tensor = torch.as_tensor(sequence)
  if sequence_tensor.dim() == 1:
    sequence_tensor = sequence_tensor.unsqueeze(-1)
  sequence_tensor = sequence_tensor.to(dtype=torch.float32)

  if length is None:
    length_value = int(sequence_tensor.shape[0])
  else:
    length_value = int(torch.as_tensor(length).item())

  label_value = int(torch.as_tensor(label).item())

  return sequence_tensor, length_value, label_value


def collate_rnn_batch(batch, padding_value=0.0):
  """Pads a batch of variable length sequences for PyTorch training."""
  if not batch:
    raise ValueError('Batches must contain at least one example.')

  sequences = []
  lengths = []
  labels = []

  for example in batch:
    sequence_tensor, length_value, label_value = _extract_example_components(
        example)
    sequences.append(sequence_tensor)
    lengths.append(max(length_value, 0))
    labels.append(label_value)

  max_length = max(lengths) if lengths else 0
  feature_dim = sequences[0].shape[-1]

  padded = torch.full(
      (len(sequences), max_length, feature_dim),
      fill_value=padding_value,
      dtype=torch.float32)
  mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)

  for i, (sequence_tensor, length_value) in enumerate(zip(sequences, lengths)):
    if length_value == 0:
      continue
    padded[i, :length_value] = sequence_tensor[:length_value]
    mask[i, :length_value] = True

  batch_dict = {
      'sequences': padded,
      'lengths': torch.tensor(lengths, dtype=torch.long),
      'mask': mask,
      'labels': torch.tensor(labels, dtype=torch.long),
  }
  return batch_dict


def _ensure_dataloader(
    data: Optional[Union[DatasetLike, Iterable]],
    batch_size: int,
    shuffle: bool,
    collate_fn) -> Optional[torch_data.DataLoader]:
  """Creates a DataLoader if one was not provided."""
  if data is None:
    return None

  if isinstance(data, torch_data.DataLoader):
    return data

  dataset_obj = data
  if isinstance(data, Iterable) and not isinstance(
      data, (torch_data.Dataset, Sequence)):
    dataset_obj = list(data)

  return torch_data.DataLoader(
      dataset_obj,
      batch_size=batch_size,
      shuffle=shuffle,
      collate_fn=collate_fn)


def _infer_input_dim(loader: torch_data.DataLoader) -> int:
  """Infers the feature dimension from the first batch in the loader."""
  iterator = iter(loader)
  try:
    sample_batch = next(iterator)
  except StopIteration as exc:
    raise ValueError('Training data is empty.') from exc

  sequences = sample_batch['sequences']
  if sequences.dim() != 3:
    raise ValueError(
        'Expected padded sequences with rank 3, received shape %r.' %
        (tuple(sequences.shape),))

  return sequences.shape[-1]


def _run_epoch(
    model: rnn.RnnClassifier,
    loader: torch_data.DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module) -> Tuple[float, float]:
  """Runs a single train or evaluation epoch."""
  is_training = optimizer is not None
  if is_training:
    model.train()
  else:
    model.eval()

  total_loss = 0.0
  total_correct = 0
  total_examples = 0

  for batch in loader:
    sequences = batch['sequences'].to(device)
    lengths = batch['lengths'].to(device)
    labels = batch['labels'].to(device)
    mask = batch.get('mask')
    if mask is not None:
      mask = mask.to(device)

    if is_training:
      optimizer.zero_grad()
      logits = model(sequences, lengths, mask=mask)
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()
    else:
      with torch.no_grad():
        logits = model(sequences, lengths, mask=mask)
        loss = criterion(logits, labels)

    total_loss += loss.item() * labels.size(0)
    predictions = logits.argmax(dim=1)
    total_correct += (predictions == labels).sum().item()
    total_examples += labels.size(0)

  avg_loss = total_loss / total_examples if total_examples else 0.0
  accuracy = total_correct / total_examples if total_examples else 0.0

  return avg_loss, accuracy


def train_rnn_classifier(
    params,
    train_data: Union[DatasetLike, Iterable],
    validation_data: Optional[Union[DatasetLike, Iterable]] = None,
    num_epochs: Optional[int] = None,
    device: Optional[torch.device] = None,
    collate_fn=collate_rnn_batch):
  """Trains the PyTorch RNN classifier using RMSProp."""

  if num_epochs is None:
    num_epochs = params.get('num_epochs', 1)

  batch_size = params['batch_size']
  train_loader = _ensure_dataloader(train_data, batch_size, True, collate_fn)
  if train_loader is None:
    raise ValueError('Training data must be provided.')

  validation_loader = _ensure_dataloader(
      validation_data, batch_size, False, collate_fn)

  input_dim = params.get('input_size') or params.get('residue_encoding_size')
  if input_dim is None:
    input_dim = _infer_input_dim(train_loader)

  model = rnn.RnnClassifier(
      input_size=input_dim,
      hidden_size=params['num_units'],
      num_layers=params['num_layers'],
      num_classes=params['num_classes'],
      dropout=params.get('dropout', 0.0))

  device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  optimizer = torch.optim.RMSprop(
      model.parameters(), lr=params['learning_rate'])
  criterion = nn.CrossEntropyLoss()

  history = {
      'train_loss': [],
      'train_accuracy': [],
  }
  if validation_loader is not None:
    history['val_loss'] = []
    history['val_accuracy'] = []

  for _ in range(num_epochs):
    train_loss, train_accuracy = _run_epoch(
        model, train_loader, device, optimizer, criterion)
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)

    if validation_loader is not None:
      val_loss, val_accuracy = _run_epoch(
          model, validation_loader, device, None, criterion)
      history['val_loss'].append(val_loss)
      history['val_accuracy'].append(val_accuracy)

  return model, history


def main(_):
  _ensure_tensorflow()
  _ensure_dataset_utils()

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
  if FLAGS.model == 'cnn':
    _ensure_cnn()
    default_hparams = _DEFAULT_HPARAMS_CNN
    model_fn = functools.partial(cnn.cnn_model_fn, refs=None)
    default_hparams['residue_encoding_size'] = seq_encoder.encoding_size
    default_hparams['seq_encoding_length'] = (len(FLAGS.ref_seq) + 1) * 2
  elif FLAGS.model == 'rnn':
    default_hparams = _DEFAULT_HPARAMS_RNN
    model_fn = rnn.rnn_model_fn
  elif FLAGS.model == 'logistic':
    _ensure_lr()
    default_hparams = _DEFAULT_HPARAMS_LOGISTIC
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
