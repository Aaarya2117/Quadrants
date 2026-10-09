"""Shared network definition: the one MLP every module imports.

Model A and Model B must use this exact class so a saved state_dict loads into
either. The layer names (fc1, fc2) are part of the contract: the .pt files, the
runtime loader and the MATLAB weight heatmap all read them.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 2


class MLP(nn.Module):
    """2 inputs -> 16 hidden units (Tanh) -> 2 outputs. Returns raw scores (logits)."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))
