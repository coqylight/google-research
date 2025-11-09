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
"""RNN model for learning the packaging phenotype from mutant sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

tf = None
tf_estimator = None

import torch
from torch import nn
from torch.nn import utils as nn_utils


def _load_tf():
  """Dynamically imports TensorFlow when needed."""
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


def rnn_model_fn(features, labels, mode, params):
  """RNN tf.estimator.Estimator model_fn definition.

  Args:
    features: ({str: Tensor}) The feature tensors provided by the input_fn.
    labels: (Tensor) The labels tensor provided by the input_fn.
    mode: (tf.estimator.ModeKeys) The invocation mode of the model.
    params: (dict) Model configuration parameters.
  Returns:
    (tf.estimator.EstimatorSpec) Model specification.
  """
  _load_tf()
  # Support both dict-based and HParams-based params.
  if not isinstance(params, dict):
    params = params.values()

  logits_train = build_rnn_inference_subgraph(
      features, reuse=False, params=params)
  logits_test = build_rnn_inference_subgraph(
      features, reuse=True, params=params)

  pred_labels = tf.argmax(logits_test, axis=1)
  pred_probas = tf.nn.softmax(logits_test)

  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'label': pred_labels,
            'proba': pred_probas,
        },
    )

  # Note: labels=None when mode==PREDICT (see tf.estimator API).
  one_hot_labels = tf.one_hot(labels, params['num_classes'])

  if mode == tf_estimator.ModeKeys.TRAIN:
    loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_train, labels=one_hot_labels))
    tf.summary.scalar('loss_train', loss_train)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss_train, global_step=tf.train.get_global_step())

    return tf_estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss_train,
    )

  accuracy = tf.metrics.accuracy(
      labels=labels,
      predictions=pred_labels)
  precision = tf.metrics.precision(
      labels=labels,
      predictions=pred_labels)
  recall = tf.metrics.recall(
      labels=labels,
      predictions=pred_labels)
  loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=logits_test, labels=one_hot_labels))
  tf.summary.scalar('loss_test', loss_test)

  return tf_estimator.EstimatorSpec(
      mode=mode,
      loss=loss_test,
      eval_metric_ops={
          'accuracy': accuracy,
          'precision': precision,
          'recall': recall,
      }
  )


def build_rnn_inference_subgraph(features, reuse, params):
  """Builds the inference subgraph for the RNN.

  Args:
    features: ({str: Tensor}) The feature tensors provided by the input_fn.
    reuse: (bool) Should the variables declared be reused?
    params: (dict) Model configuration parameters.
  Returns:
    (Tensor) A reference to the logits tensor for the inference subgraph.
  """
  _load_tf()
  with tf.variable_scope('inference', reuse=reuse):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(
            num_units=params['num_units'],
            state_is_tuple=True) for _ in range(params['num_layers'])
    ])

    initial_state = cell.zero_state(params['batch_size'], tf.float32)

    outputs, unused_state = tf.nn.dynamic_rnn(
        cell,
        features['sequence'],
        sequence_length=features['sequence_length'],
        initial_state=initial_state,
        dtype=tf.float32)
    output = tf.reduce_mean(outputs, axis=1)
    logits = tf.layers.dense(
        output, params['num_classes'], activation=None, use_bias=True)

    return logits


class RnnClassifier(nn.Module):
  """PyTorch implementation of the AAV sequence classifier."""

  def __init__(
      self,
      input_size,
      hidden_size,
      num_layers,
      num_classes,
      dropout=0.0):
    super().__init__()
    self.lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0.0,
        batch_first=True)
    self.classifier = nn.Linear(hidden_size, num_classes)

  def forward(self, sequences, lengths, mask=None):
    """Runs the classifier on a batch of padded sequences."""
    if mask is not None:
      mask = mask.to(sequences.device)
      derived_lengths = mask.to(dtype=torch.long).sum(dim=1)
      if lengths is None:
        lengths = derived_lengths
      else:
        lengths = torch.minimum(
            lengths.to(device=sequences.device, dtype=torch.long),
            derived_lengths)

    if lengths is None:
      lengths = torch.full(
          (sequences.size(0),),
          fill_value=sequences.size(1),
          dtype=torch.long,
          device=sequences.device)

    lengths_cpu = lengths.to(dtype=torch.long).cpu()
    packed = nn_utils.rnn.pack_padded_sequence(
        sequences,
        lengths_cpu,
        batch_first=True,
        enforce_sorted=False)
    packed_outputs, _ = self.lstm(packed)
    outputs, _ = nn_utils.rnn.pad_packed_sequence(
        packed_outputs,
        batch_first=True)

    max_time = outputs.size(1)
    if mask is None:
      time_indices = torch.arange(max_time, device=outputs.device)
      mask = time_indices.unsqueeze(0) < lengths.unsqueeze(1)
    else:
      mask = mask[:, :max_time]

    mask = mask.to(outputs.dtype).unsqueeze(-1)
    summed = (outputs * mask).sum(dim=1)
    valid_lengths = mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / valid_lengths
    logits = self.classifier(pooled)
    return logits
