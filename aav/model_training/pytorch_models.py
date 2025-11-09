# coding=utf-8
"""PyTorch model definitions for AAV training."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class LogisticRegressionClassifier(nn.Module):
  """Simple logistic regression classifier operating on flattened features."""

  def __init__(self, num_features: int, num_classes: int):
    """Initialises the classifier.

    Args:
      num_features: Total number of input features.
      num_classes: Number of output classes for the classification task.
    """
    super().__init__()
    self._num_features = num_features
    self.linear = nn.Linear(num_features, num_classes)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Applies the linear projection after flattening the inputs."""
    if inputs.dim() != 2:
      batch_size = inputs.shape[0]
      inputs = inputs.reshape(batch_size, -1)
    return self.linear(inputs)

  @property
  def input_shape(self) -> Tuple[int, ...]:
    return (self._num_features,)
