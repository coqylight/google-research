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
"""PyTorch model definitions mirroring the TensorFlow Estimator models."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class RNNClassifier(nn.Module):
  """Multi-layer LSTM classifier with mean-pooled outputs."""

  def __init__(
      self,
      input_size: int,
      num_classes: int,
      num_layers: int,
      hidden_size: int,
      dropout: float = 0.0,
  ):
    super().__init__()
    self._lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    self._classifier = nn.Linear(hidden_size, num_classes)

  def forward(self, sequence: torch.Tensor, sequence_length: Optional[torch.Tensor] = None):
    if sequence_length is not None:
      packed = nn.utils.rnn.pack_padded_sequence(
          sequence,
          sequence_length.cpu(),
          batch_first=True,
          enforce_sorted=False)
      outputs, _ = self._lstm(packed)
      outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
      max_len = outputs.shape[1]
      lengths = sequence_length.to(sequence.device).unsqueeze(-1).float()
      mask = torch.arange(max_len, device=sequence.device).unsqueeze(0)
      mask = mask < sequence_length.unsqueeze(1)
      mask = mask.unsqueeze(-1).float()
      summed = (outputs * mask).sum(dim=1)
      pooled = summed / lengths.clamp(min=1.0)
    else:
      outputs, _ = self._lstm(sequence)
      pooled = outputs.mean(dim=1)
    return self._classifier(pooled)


class CNNClassifier(nn.Module):
  """1-D convolutional classifier for mutation encodings."""

  def __init__(
      self,
      residue_encoding_size: int,
      seq_encoding_length: int,
      num_classes: int,
      conv_depth: int,
      conv_depth_multiplier: int,
      conv_width: int,
      pool_width: int,
      fc_size: int,
      fc_size_multiplier: float,
  ):
    super().__init__()
    self._residue_encoding_size = residue_encoding_size
    self._seq_encoding_length = seq_encoding_length
    self._conv1 = nn.Conv1d(
        in_channels=residue_encoding_size,
        out_channels=conv_depth,
        kernel_size=conv_width,
        padding=conv_width // 2)
    self._bn1 = nn.BatchNorm1d(conv_depth)
    self._conv2 = nn.Conv1d(
        in_channels=conv_depth,
        out_channels=conv_depth * conv_depth_multiplier,
        kernel_size=conv_width,
        padding=conv_width // 2)
    self._bn2 = nn.BatchNorm1d(conv_depth * conv_depth_multiplier)
    self._pool = nn.MaxPool1d(kernel_size=pool_width, stride=pool_width)

    flattened_length = (
        (seq_encoding_length // pool_width // pool_width)
        * (conv_depth * conv_depth_multiplier))
    self._fc1 = nn.Linear(
        flattened_length,
        fc_size)
    self._bn_fc1 = nn.BatchNorm1d(fc_size)
    self._fc2 = nn.Linear(int(fc_size), int(fc_size * fc_size_multiplier))
    self._bn_fc2 = nn.BatchNorm1d(int(fc_size * fc_size_multiplier))
    self._classifier = nn.Linear(int(fc_size * fc_size_multiplier), num_classes)

  def forward(self, sequence: torch.Tensor, sequence_length: Optional[torch.Tensor] = None):
    batch = sequence.shape[0]
    seq = sequence.view(batch, -1, self._residue_encoding_size)
    seq = seq.permute(0, 2, 1)
    x = F.relu(self._bn1(self._conv1(seq)))
    x = self._pool(x)
    x = F.relu(self._bn2(self._conv2(x)))
    x = self._pool(x)
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self._bn_fc1(self._fc1(x)))
    x = F.relu(self._bn_fc2(self._fc2(x)))
    return self._classifier(x)


class LogisticRegression(nn.Module):
  """Single linear layer over flattened mutation encodings."""

  def __init__(
      self,
      residue_encoding_size: int,
      seq_encoding_length: int,
      num_classes: int,
  ):
    super().__init__()
    input_dim = residue_encoding_size * seq_encoding_length
    self._linear = nn.Linear(input_dim, num_classes)

  def forward(self, sequence: torch.Tensor, sequence_length: Optional[torch.Tensor] = None):
    batch = sequence.shape[0]
    flattened = sequence.view(batch, -1)
    return self._linear(flattened)
