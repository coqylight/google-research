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
#
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
"""PyTorch implementation of the AAV CNN classifier."""

from __future__ import annotations

import copy
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import nn


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
    'positive_class': 1,
}


def _compute_pooled_length(sequence_length: int, pool_width: int) -> int:
    """Computes the sequence length after two max-pooling layers."""
    length = sequence_length
    for _ in range(2):
        if length <= 0:
            return 1
        length = (length - pool_width) // pool_width + 1
        if length <= 0:
            length = 1
    return length


class CnnClassifier(nn.Module):
    """CNN classifier that mirrors the TensorFlow implementation."""

    def __init__(self, **hparams: float) -> None:
        super().__init__()
        params = copy.deepcopy(_DEFAULT_HPARAMS_CNN)
        params.update(hparams)

        seq_length = params['seq_encoding_length']
        residue_size = params['residue_encoding_size']
        num_classes = params['num_classes']
        if seq_length is None or residue_size is None:
            raise ValueError('Both seq_encoding_length and residue_encoding_size must be provided.')
        if num_classes is None:
            raise ValueError('num_classes must be provided.')

        self.hparams = params
        self._positive_class = params.get('positive_class', num_classes - 1)

        conv_depth = params['conv_depth']
        conv_multiplier = params['conv_depth_multiplier']
        conv_width = params['conv_width']
        pool_width = params['pool_width']
        fc_size = params['fc_size']
        fc_size_multiplier = params['fc_size_multiplier']

        conv2_depth = conv_depth * conv_multiplier
        pooled_length = _compute_pooled_length(seq_length, pool_width)
        fc2_size = int(fc_size * fc_size_multiplier)

        padding = conv_width // 2

        self.conv1 = nn.Conv1d(
            in_channels=residue_size,
            out_channels=conv_depth,
            kernel_size=conv_width,
            padding=padding,
        )
        self.conv1_bn = nn.BatchNorm1d(conv_depth)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_width, stride=pool_width)

        self.conv2 = nn.Conv1d(
            in_channels=conv_depth,
            out_channels=conv2_depth,
            kernel_size=conv_width,
            padding=padding,
        )
        self.conv2_bn = nn.BatchNorm1d(conv2_depth)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_width, stride=pool_width)

        flattened_size = conv2_depth * pooled_length
        self.fc1 = nn.Linear(flattened_size, fc_size)
        self.fc1_bn = nn.BatchNorm1d(fc_size)

        self.fc2 = nn.Linear(fc_size, fc2_size)
        self.fc2_bn = nn.BatchNorm1d(fc2_size)

        self.output = nn.Linear(fc2_size, num_classes)

    @property
    def positive_class(self) -> int:
        return self._positive_class

    def forward(self, inputs: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the logits for a batch of inputs."""
        if isinstance(inputs, dict):
            x = inputs['sequence']
        else:
            x = inputs

        if x.dim() == 2:
            x = x.view(
                x.size(0),
                self.hparams['seq_encoding_length'],
                self.hparams['residue_encoding_size'])
        elif x.dim() == 3:
            if x.size(1) == self.hparams['seq_encoding_length'] and x.size(2) == self.hparams['residue_encoding_size']:
                pass
            elif x.size(1) == self.hparams['residue_encoding_size'] and x.size(2) == self.hparams['seq_encoding_length']:
                x = x.permute(0, 2, 1)
            else:
                raise ValueError('Unexpected input shape: %r' % (tuple(x.size()),))
        else:
            raise ValueError('Expected input tensor with 2 or 3 dimensions, got %d.' % x.dim())

        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = torch.relu(self.fc2(x))
        x = self.fc2_bn(x)
        logits = self.output(x)
        return logits

    def predict_proba(self, inputs: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns class probabilities for the given inputs."""
        logits = self.forward(inputs)
        return torch.softmax(logits, dim=-1)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_class: int = 1,
) -> Dict[str, float]:
    """Computes accuracy, precision, and recall for predictions."""
    preds = torch.argmax(logits.detach(), dim=-1)
    preds = preds.to('cpu')
    labels_cpu = labels.detach().to('cpu')

    accuracy = (preds == labels_cpu).float().mean().item()

    positive_mask = labels_cpu == positive_class
    predicted_positive = preds == positive_class
    true_positive = torch.sum(positive_mask & predicted_positive).item()
    false_positive = torch.sum(~positive_mask & predicted_positive).item()
    false_negative = torch.sum(positive_mask & ~predicted_positive).item()

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (true_positive / precision_denominator) if precision_denominator else 0.0
    recall = (true_positive / recall_denominator) if recall_denominator else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    }


def make_training_step(
    model: nn.Module,
    learning_rate: Optional[float] = None,
    device: Optional[torch.device] = None,
    positive_class: Optional[int] = None,
    metric_hooks: Optional[Iterable[Callable[[Dict[str, float]], None]]] = None,
):
    """Builds a single-step training function for the classifier."""
    if learning_rate is None:
        learning_rate = model.hparams.get('learning_rate', 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    positive_label = positive_class if positive_class is not None else getattr(model, 'positive_class', 1)

    def step(batch):
        features, labels = batch
        if isinstance(features, dict):
            features = features['sequence']
        if device is not None:
            features = features.to(device)
            labels = labels.to(device)

        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        metrics = compute_classification_metrics(logits, labels, positive_label)
        metrics['loss'] = loss.item()
        metrics['batch_size'] = int(labels.size(0))

        if metric_hooks:
            for hook in metric_hooks:
                hook(metrics)
        return metrics

    return step, optimizer


__all__ = [
    'CnnClassifier',
    'compute_classification_metrics',
    'make_training_step',
]
