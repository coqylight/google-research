# coding=utf-8
"""Unified PyTorch training and evaluation entry point for AAV models."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tensorflow.core.example import example_pb2
from tensorflow.python.lib.io import tf_record

from .pytorch_models import LogisticRegressionClassifier
from ..util import dataset_utils


DEFAULT_NUM_CLASSES = 2
DEFAULT_BATCH_SIZE = 25
DEFAULT_LEARNING_RATE = 0.01  # Matches _DEFAULT_HPARAMS_LOGISTIC in train.py.
POSITIVE_LABEL = 1


def _tf_record_iterator(filepaths: Sequence[str]) -> Iterable[example_pb2.Example]:
  for filepath in filepaths:
    for raw_record in tf_record.tf_record_iterator(filepath):
      example = example_pb2.Example()
      example.ParseFromString(raw_record)
      yield example


def _resolve_filepaths(paths: Sequence[str]) -> List[str]:
  resolved = []
  for path in paths:
    if os.path.isdir(path):
      resolved.extend(
          os.path.join(path, entry)
          for entry in os.listdir(path)
          if entry.endswith('.tfrecord'))
    else:
      resolved.append(path)
  return resolved


def _decode_example(example: example_pb2.Example) -> Tuple[np.ndarray, int]:
  mutation_sequence = example.features.feature['mutation_sequence'].bytes_list.value[0]
  label = example.features.feature['is_viable'].int64_list.value[0]
  encoded = dataset_utils.ONEHOT_FIXEDLEN_MUTATION_ENCODER.encode(
      mutation_sequence.decode('utf-8'))
  return encoded.astype(np.float32), int(label)


def _load_fixed_length_dataset(filepaths: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
  features: List[np.ndarray] = []
  labels: List[int] = []
  for example in _tf_record_iterator(_resolve_filepaths(filepaths)):
    encoded, label = _decode_example(example)
    features.append(encoded)
    labels.append(label)

  if not features:
    raise ValueError('No TFRecord examples found in %s' % (filepaths,))

  feature_tensor = torch.from_numpy(np.stack(features)).float()
  label_tensor = torch.tensor(labels, dtype=torch.long)
  return feature_tensor, label_tensor


class FixedLengthSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):

  def __init__(self, filepaths: Sequence[str]):
    self._features, self._labels = _load_fixed_length_dataset(filepaths)

  def __len__(self) -> int:
    return self._labels.shape[0]

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return self._features[idx], self._labels[idx]

  @property
  def num_features(self) -> int:
    return int(self._features[0].numel())


def configure_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
  return torch.optim.SGD(model.parameters(), lr=learning_rate)


def compute_binary_metrics(
    logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
  predictions = torch.argmax(logits, dim=1)
  correct = (predictions == labels).float()
  accuracy = correct.mean().item()

  positive_predictions = predictions == POSITIVE_LABEL
  positive_labels = labels == POSITIVE_LABEL
  true_positive = (positive_predictions & positive_labels).sum().item()
  predicted_positive = positive_predictions.sum().item()
  actual_positive = positive_labels.sum().item()

  precision = true_positive / predicted_positive if predicted_positive else 0.0
  recall = true_positive / actual_positive if actual_positive else 0.0

  return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
  }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module) -> Dict[str, float]:
  model.eval()
  total_loss = 0.0
  total_examples = 0
  all_logits: List[torch.Tensor] = []
  all_labels: List[torch.Tensor] = []
  with torch.no_grad():
    for features, labels in data_loader:
      features = features.to(device)
      labels = labels.to(device)
      logits = model(features)
      loss = loss_fn(logits, labels)
      batch_size = labels.size(0)
      total_loss += loss.item() * batch_size
      total_examples += batch_size
      all_logits.append(logits.cpu())
      all_labels.append(labels.cpu())

  aggregated_logits = torch.cat(all_logits, dim=0)
  aggregated_labels = torch.cat(all_labels, dim=0)
  metrics = compute_binary_metrics(aggregated_logits, aggregated_labels)
  metrics['loss'] = total_loss / max(total_examples, 1)
  return metrics


def train_and_evaluate(args: argparse.Namespace) -> None:
  device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

  train_dataset = FixedLengthSequenceDataset(args.train_path)
  validation_dataset = FixedLengthSequenceDataset(args.validation_path)

  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True)
  validation_loader = DataLoader(
      validation_dataset,
      batch_size=args.batch_size,
      shuffle=False)

  num_features = train_dataset.num_features
  model = LogisticRegressionClassifier(num_features, args.num_classes).to(device)

  optimizer = configure_optimizer(model, args.learning_rate)
  loss_fn = nn.CrossEntropyLoss()

  best_metric = float('-inf')
  best_state = None

  for epoch in range(args.max_epochs):
    model.train()
    for step, (features, labels) in enumerate(train_loader):
      features = features.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      logits = model(features)
      loss = loss_fn(logits, labels)
      loss.backward()
      optimizer.step()
      if (step + 1) % args.log_every_steps == 0:
        print(f'Epoch {epoch+1} Step {step+1} Loss {loss.item():.4f}')

    metrics = evaluate(model, validation_loader, device, loss_fn)
    monitored_value = metrics[args.eval_metric]
    print(f'Epoch {epoch+1}: {json.dumps(metrics)}')

    if monitored_value > best_metric:
      best_metric = monitored_value
      best_state = {
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'metrics': metrics,
          'epoch': epoch + 1,
      }

  if args.model_dir and best_state is not None:
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.model_dir, 'best_model.pt'))
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
      json.dump(best_state['metrics'], f)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', default='logistic', choices=['logistic'])
  parser.add_argument('--seq_encoder', default='fixedlen-id', choices=['fixedlen-id'])
  parser.add_argument('--train_path', nargs='+', required=True)
  parser.add_argument('--validation_path', nargs='+', required=True)
  parser.add_argument('--model_dir', default=None)
  parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
  parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
  parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES)
  parser.add_argument('--max_epochs', type=int, default=10)
  parser.add_argument('--log_every_steps', type=int, default=50)
  parser.add_argument('--eval_metric', default='precision', choices=['precision', 'recall', 'accuracy'])
  parser.add_argument('--use_gpu', action='store_true')
  return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
  args = parse_args(argv)
  train_and_evaluate(args)


if __name__ == '__main__':
  import sys
  main(sys.argv[1:])
