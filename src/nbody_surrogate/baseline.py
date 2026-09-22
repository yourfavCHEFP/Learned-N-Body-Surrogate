"""Simple MLP baseline for N-body acceleration prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Multi-layer perceptron that predicts accelerations from flattened states.

    Args:
        input_dim: Number of input features (N_bodies × 7)
        hidden_dim: Number of neurons in each hidden layer
        output_dim: Number of output features (N_bodies × 3)
        n_layers: Number of hidden layers (default: 2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 9,
        n_layers: int = 2,
    ):
        super().__init__()

        # Build a list of layers for the MLP
        layers = []

        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Middle layers: hidden → hidden
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Final layer: hidden → output (no activation!)
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Wrap everything in a Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Predictions of shape (batch_size, output_dim) or (output_dim,)
        """
        return self.network(x)
