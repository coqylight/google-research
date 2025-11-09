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
"""PyTorch training driver for the AAV models."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils import data as torch_data

from ..model_training import train_utils
from ..util import dataset_utils
from . import dataset as torch_dataset
from . import models as torch_models

_DEFAULT_HPARAMS_RNN = {
    'batch_size': 25,
    'learning_rate': 0.01,
    'num_classes': 2,
    'num_layers': 2,
    'num_units': 100,
    'dropout': 0.0,
}

_DEFAULT_HPARAMS_CNN = {
    'batch_size': 25,
    'learning_rate': 0.001,
    'num_classes': 2,
    'residue_encoding_size': None,
    'seq_encoding_length': None,
    'pool_width': 2,
    'conv_depth': 12,
    'conv_depth_multiplier': 2,
    'conv_width': 7,
    'fc_size': 128,
    'fc_size_multiplier': 0.5,
}

_DEFAULT_HPARAMS_LOGISTIC = {
    'batch_size': 25,
    'learning_rate': 0.01,
    'num_classes': 2,
    'residue_encoding_size': None,
    'seq_encoding_length': None,
}


def _convert_value(value: str):
  lowered = value.lower()
  if lowered in ('true', 'false'):
    return lowered == 'true'
  try:
    return int(value)
  except ValueError:
    pass
  try:
    return float(value)
  except ValueError:
    return value


def parse_hparams(overrides: Optional[str]) -> Dict[str, object]:
  params = {}
  if not overrides:
    return params
  assignments = [v.strip() for v in overrides.split(',') if v.strip()]
  for assignment in assignments:
    if '=' not in assignment:
      raise ValueError('Invalid hparam override: %s' % assignment)
    key, value = assignment.split('=', 1)
    params[key.strip()] = _convert_value(value.strip())
  return params


class MetricsTracker(object):
  """Aggregates metrics across batches."""

  def __init__(self, positive_label: int = 1):
    self._positive_label = positive_label
    self.reset()

  def reset(self):
    self._total_loss = 0.0
    self._total_examples = 0
    self._correct = 0
    self._true_positive = 0
    self._false_positive = 0
    self._false_negative = 0

  def update(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
    batch_size = labels.shape[0]
    self._total_examples += batch_size
    self._total_loss += float(loss.item()) * batch_size

    preds = torch.argmax(logits, dim=1)
    self._correct += int((preds == labels).sum().item())

    pos = self._positive_label
    true_positive = ((preds == pos) & (labels == pos)).sum().item()
    false_positive = ((preds == pos) & (labels != pos)).sum().item()
    false_negative = ((preds != pos) & (labels == pos)).sum().item()

    self._true_positive += int(true_positive)
    self._false_positive += int(false_positive)
    self._false_negative += int(false_negative)

  def compute(self) -> Dict[str, float]:
    if not self._total_examples:
      return {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    loss = self._total_loss / self._total_examples
    accuracy = self._correct / float(self._total_examples)
    precision_den = self._true_positive + self._false_positive
    recall_den = self._true_positive + self._false_negative
    precision = (self._true_positive / precision_den) if precision_den else 0.0
    recall = (self._true_positive / recall_den) if recall_den else 0.0
    return {
      'loss': loss,
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
    }


def _build_model(model_name: str, hparams: Dict[str, object]):
  if model_name == 'rnn':
    return torch_models.RNNClassifier(
        input_size=hparams['residue_encoding_size'],
        num_classes=hparams['num_classes'],
        num_layers=hparams['num_layers'],
        hidden_size=hparams['num_units'],
        dropout=hparams.get('dropout', 0.0))
  if model_name == 'cnn':
    return torch_models.CNNClassifier(
        residue_encoding_size=hparams['residue_encoding_size'],
        seq_encoding_length=hparams['seq_encoding_length'],
        num_classes=hparams['num_classes'],
        conv_depth=hparams['conv_depth'],
        conv_depth_multiplier=hparams['conv_depth_multiplier'],
        conv_width=hparams['conv_width'],
        pool_width=hparams['pool_width'],
        fc_size=hparams['fc_size'],
        fc_size_multiplier=hparams['fc_size_multiplier'])
  if model_name == 'logistic':
    return torch_models.LogisticRegression(
        residue_encoding_size=hparams['residue_encoding_size'],
        seq_encoding_length=hparams['seq_encoding_length'],
        num_classes=hparams['num_classes'])
  raise ValueError('Unsupported model: %s' % model_name)


def _build_optimizer(model: torch.nn.Module, model_name: str, hparams: Dict[str, object]):
  lr = hparams['learning_rate']
  if model_name == 'rnn':
    return torch.optim.RMSprop(model.parameters(), lr=lr)
  if model_name == 'cnn':
    return torch.optim.Adam(model.parameters(), lr=lr)
  if model_name == 'logistic':
    return torch.optim.Adagrad(model.parameters(), lr=lr)
  raise ValueError('Unsupported model: %s' % model_name)


def _build_scheduler(optimizer: torch.optim.Optimizer, hparams: Dict[str, object]):
  step_size = hparams.get('lr_step_size')
  gamma = hparams.get('lr_gamma')
  if step_size and gamma:
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(step_size), gamma=float(gamma))
  return None


def _select_encoder(seq_encoder: str):
  if seq_encoder == 'varlen-id':
    return dataset_utils.ONEHOT_VARLEN_SEQUENCE_ENCODER, 'sequence', True
  if seq_encoder == 'fixedlen-id':
    return dataset_utils.ONEHOT_FIXEDLEN_MUTATION_ENCODER, 'mutation_sequence', False
  raise ValueError('Unsupported sequence encoder: %s' % seq_encoder)


def _load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    early_stopper: train_utils.EarlyStopper,
    checkpoint_path: str,
    device: torch.device):
  checkpoint = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if scheduler and checkpoint.get('scheduler_state_dict'):
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  restored_early = checkpoint.get('early_stopper')
  if restored_early is not None:
    early_stopper._evals = restored_early._evals  # pylint: disable=protected-access
    early_stopper._best_so_far = restored_early._best_so_far  # pylint: disable=protected-access
    early_stopper._num_evals_since_best = restored_early._num_evals_since_best  # pylint: disable=protected-access
  return checkpoint.get('global_step', 0), checkpoint.get('best_metric')


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    early_stopper: train_utils.EarlyStopper,
    model_dir: str,
    global_step: int,
    best_metric: Optional[float],
    is_best: bool):
  payload = {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
      'early_stopper': early_stopper,
      'global_step': global_step,
      'best_metric': best_metric,
  }
  torch.save(payload, os.path.join(model_dir, 'checkpoint.pt'))
  if is_best and best_metric is not None:
    torch.save(payload, os.path.join(model_dir, 'best.pt'))


def _evaluate(
    model: torch.nn.Module,
    dataloader: torch_data.DataLoader,
    criterion: nn.Module,
    device: torch.device) -> Dict[str, float]:
  model.eval()
  tracker = MetricsTracker()
  with torch.no_grad():
    for features, labels in dataloader:
      sequence = features['sequence'].to(device)
      lengths = features.get('sequence_length')
      if lengths is not None:
        lengths = lengths.to(device)
      labels = labels.to(device)
      logits = model(sequence, sequence_length=lengths)
      loss = criterion(logits, labels)
      tracker.update(loss, logits, labels)
  return tracker.compute()


def _format_metrics(name: str, step: int, metrics: Dict[str, float]) -> str:
  return (
      f"{name} step {step}: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, "
      f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")


def run_training(args):
  device = torch.device(args.device)
  logging.info('Using device: %s', device)
  model_dir = args.model_dir
  os.makedirs(model_dir, exist_ok=True)

  default_hparams = {
      'rnn': dict(_DEFAULT_HPARAMS_RNN),
      'cnn': dict(_DEFAULT_HPARAMS_CNN),
      'logistic': dict(_DEFAULT_HPARAMS_LOGISTIC),
  }[args.model]
  overrides = parse_hparams(args.hparams)
  default_hparams.update(overrides)

  encoder, sequence_key, include_length = _select_encoder(args.seq_encoder)
  default_hparams['residue_encoding_size'] = encoder.encoding_size
  if not include_length:
    default_hparams['seq_encoding_length'] = (len(args.ref_seq) + 1) * 2

  model = _build_model(args.model, default_hparams).to(device)
  optimizer = _build_optimizer(model, args.model, default_hparams)
  scheduler = _build_scheduler(optimizer, default_hparams)
  criterion = nn.CrossEntropyLoss()

  train_dataset = torch_dataset.SequenceDataset(
      args.train_path,
      encoder=encoder,
      sequence_key=sequence_key,
      include_length=include_length,
      cache_path=args.train_cache)
  valid_dataset = torch_dataset.SequenceDataset(
      args.validation_path,
      encoder=encoder,
      sequence_key=sequence_key,
      include_length=include_length,
      cache_path=args.validation_cache)

  collate_fn = torch_dataset.collate_batch
  train_loader = torch_data.DataLoader(
      train_dataset,
      batch_size=default_hparams['batch_size'],
      shuffle=True,
      drop_last=True,
      collate_fn=collate_fn,
      num_workers=args.num_workers)
  valid_loader = torch_data.DataLoader(
      valid_dataset,
      batch_size=default_hparams['batch_size'],
      shuffle=False,
      drop_last=False,
      collate_fn=collate_fn,
      num_workers=args.num_workers)

  early_stopper = train_utils.EarlyStopper(
      num_evals_to_wait=args.early_stopper_num_evals_to_wait,
      metric_key=args.eval_metric)

  global_step = 0
  best_metric = None

  if args.resume:
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
      global_step, best_metric = _load_checkpoint(
          model, optimizer, scheduler, early_stopper, checkpoint_path, device)
      logging.info('Resumed from %s at step %d', checkpoint_path, global_step)

  tracker = MetricsTracker()
  model.train()

  def maybe_log_and_eval(step: int):
    nonlocal best_metric
    metrics = tracker.compute()
    logging.info(_format_metrics('Train', step, metrics))
    tracker.reset()

    eval_metrics = _evaluate(model, valid_loader, criterion, device)
    logging.info(_format_metrics('Eval', step, eval_metrics))

    current_metric = eval_metrics[args.eval_metric]
    keep_training = early_stopper.early_stop_predicate_fn({args.eval_metric: current_metric})
    is_best = best_metric is None or current_metric > best_metric
    if is_best:
      best_metric = current_metric
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        early_stopper,
        model_dir,
        step,
        best_metric,
        is_best)
    if scheduler is not None:
      scheduler.step()
    return keep_training

  keep_training = True
  while keep_training and global_step < args.max_train_steps:
    for features, labels in train_loader:
      sequence = features['sequence'].to(device)
      lengths = features.get('sequence_length')
      if lengths is not None:
        lengths = lengths.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      logits = model(sequence, sequence_length=lengths)
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()

      tracker.update(loss.detach(), logits.detach(), labels.detach())

      global_step += 1
      if global_step % args.train_steps_per_eval == 0:
        keep_training = maybe_log_and_eval(global_step)
        model.train()
        if not keep_training or global_step >= args.max_train_steps:
          break
    else:
      continue
    break

  if global_step % args.train_steps_per_eval != 0 and global_step > 0:
    maybe_log_and_eval(global_step)

  with open(os.path.join(model_dir, 'hparams.json'), 'w') as f:
    json.dump(default_hparams, f, indent=2, sort_keys=True)


def _parse_args(argv: Optional[Iterable[str]] = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model_dir', required=True, help='Directory to store checkpoints.')
  parser.add_argument('--train_path', required=True, nargs='+', help='Training TFRecord files.')
  parser.add_argument('--validation_path', required=True, nargs='+', help='Validation TFRecord files.')
  parser.add_argument('--model', required=True, choices=['rnn', 'cnn', 'logistic'])
  parser.add_argument('--seq_encoder', required=True, choices=['varlen-id', 'fixedlen-id'])
  parser.add_argument('--hparams', default=None, help='Comma-separated list of hparam overrides.')
  parser.add_argument('--eval_metric', default='precision', choices=['precision', 'recall', 'accuracy'])
  parser.add_argument('--train_steps_per_eval', type=int, default=500)
  parser.add_argument('--max_train_steps', type=int, default=100000)
  parser.add_argument('--early_stopper_num_evals_to_wait', type=int, default=10)
  parser.add_argument('--ref_seq', default='DEEEIRTTNPVATEQYGSVSTNLQRGNR')
  parser.add_argument('--train_cache', default=None, help='Optional path to cache the encoded training set.')
  parser.add_argument('--validation_cache', default=None, help='Optional path to cache the encoded validation set.')
  parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint if available.')
  parser.add_argument('--num_workers', type=int, default=0)
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None):
  args = _parse_args(argv)
  logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
  run_training(args)


if __name__ == '__main__':
  main()
