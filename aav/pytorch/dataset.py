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
"""Dataset utilities for PyTorch training."""

from __future__ import annotations

import collections
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils import data as torch_data
import tensorflow as tf

Example = collections.namedtuple('Example', ['sequence', 'sequence_length', 'label'])


def _as_bytes_list(feature):
  return feature.bytes_list.value


def _as_int_list(feature):
  return feature.int64_list.value


def _decode_tfrecord(
    record: bytes,
    *,
    encoder,
    sequence_key: str,
    label_key: str,
    include_length: bool,
) -> Example:
  """Decodes a serialized TFRecord example into numpy arrays."""
  proto = tf.train.Example()
  proto.ParseFromString(record)
  features = proto.features.feature

  raw_sequence = _as_bytes_list(features[sequence_key])[0]
  if isinstance(raw_sequence, bytes):
    sequence_str = raw_sequence.decode('utf-8')
  else:
    sequence_str = raw_sequence

  encoded = encoder.encode(sequence_str)
  sequence_tensor = torch.as_tensor(encoded, dtype=torch.float32)
  sequence_length = int(sequence_tensor.shape[0]) if include_length else None

  label = int(_as_int_list(features[label_key])[0])
  return Example(sequence_tensor, sequence_length, label)


def _load_tfrecords(
    filepaths: Sequence[str],
    *,
    encoder,
    sequence_key: str,
    label_key: str,
    include_length: bool,
) -> List[Example]:
  """Loads and encodes TFRecord examples into memory."""
  records: List[Example] = []
  for path in filepaths:
    for record in tf.compat.v1.io.tf_record_iterator(path):
      records.append(
          _decode_tfrecord(
              record,
              encoder=encoder,
              sequence_key=sequence_key,
              label_key=label_key,
              include_length=include_length))
  return records


def _load_preprocessed(path: str) -> List[Example]:
  """Loads a preprocessed torch-saved dataset."""
  payload = torch.load(path)
  sequences = payload['sequence']
  labels = payload['label']
  lengths = payload.get('sequence_length')
  examples: List[Example] = []
  for i, seq in enumerate(sequences):
    seq_tensor = torch.as_tensor(seq, dtype=torch.float32)
    if lengths is None:
      length = None
    else:
      raw_length = lengths[i]
      length = None if raw_length is None else int(raw_length)
    label = int(labels[i])
    examples.append(Example(seq_tensor, length, label))
  return examples


class SequenceDataset(torch_data.Dataset):
  """Dataset that exposes TFRecord AAV examples to PyTorch."""

  def __init__(
      self,
      data_paths: Union[str, Sequence[str]],
      *,
      encoder,
      sequence_key: str,
      label_key: str = 'is_viable',
      include_length: bool = True,
      cache_path: Optional[str] = None,
  ):
    if isinstance(data_paths, str):
      data_paths = [data_paths]

    self._return_length = include_length
    self._cache_path = cache_path

    if cache_path and os.path.exists(cache_path):
      self._examples = _load_preprocessed(cache_path)
    else:
      # Detect TFRecord inputs by extension.
      tfrecord_paths = []
      preprocessed_paths = []
      for path in data_paths:
        if path.endswith(('.pt', '.pth')):
          preprocessed_paths.append(path)
        else:
          tfrecord_paths.append(path)

      examples: List[Example] = []
      if tfrecord_paths:
        examples.extend(
            _load_tfrecords(
                tfrecord_paths,
                encoder=encoder,
                sequence_key=sequence_key,
                label_key=label_key,
                include_length=True))
      for path in preprocessed_paths:
        examples.extend(_load_preprocessed(path))

      self._examples = examples
      if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
          os.makedirs(cache_dir)
        torch.save(
            {
                'sequence': [ex.sequence.numpy() for ex in examples],
                'sequence_length': [ex.sequence_length for ex in examples],
                'label': [ex.label for ex in examples],
            },
            cache_path)

  @property
  def return_length(self) -> bool:
    return self._return_length

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, idx: int):
    example = self._examples[idx]
    features: Dict[str, torch.Tensor] = {
        'sequence': example.sequence,
    }
    if self._return_length and example.sequence_length is not None:
      features['sequence_length'] = torch.tensor(
          example.sequence_length, dtype=torch.long)
    label_tensor = torch.tensor(example.label, dtype=torch.long)
    return features, label_tensor


def collate_batch(batch: Sequence[Tuple[Dict[str, torch.Tensor], torch.Tensor]]):
  """Collate function that respects variable-length sequences."""
  features, labels = zip(*batch)
  sequences = [feat['sequence'] for feat in features]
  labels_tensor = torch.stack(labels)

  if 'sequence_length' not in features[0]:
    sequences_tensor = torch.stack(sequences)
    return {'sequence': sequences_tensor}, labels_tensor

  lengths = torch.stack([feat['sequence_length'] for feat in features])
  max_len = int(lengths.max())
  feature_shape = sequences[0].shape[1:]
  padded = sequences[0].new_zeros((len(sequences), max_len) + feature_shape)
  for i, seq in enumerate(sequences):
    seq_len = int(lengths[i])
    padded[i, :seq_len] = seq[:seq_len]

  return {
      'sequence': padded,
      'sequence_length': lengths,
  }, labels_tensor
