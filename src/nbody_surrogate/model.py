"""Graph Neural Network for N-body acceleration prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


def build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
) -> nn.Module:
    """
    Build a simple multi-layer perceptron.

    Args:
        input_dim: Number of input features
        hidden_dim: Number of neurons in hidden layers
        output_dim: Number of output features
        num_layers: Number of hidden layers (default: 2)

    Returns:
        Sequential MLP module
    """
    layers = []

    # First layer: input → hidden
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())

    # Middle layers: hidden → hidden
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())

    # Output layer: hidden → output (no activation)
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


class InteractionLayer(MessagePassing):
    """
    Single message passing layer for N-body interactions.

    Implements one round of:
    1. Compute messages along edges (edge model)
    2. Aggregate messages at nodes (sum)
    3. Update node features (node model)
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        """
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for MLPs
        """
        super().__init__(aggr="add")

        self.edge_mlp = build_mlp(
            input_dim=node_dim + node_dim + edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )

        self.node_mlp = build_mlp(
            input_dim=node_dim + hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=node_dim,
            num_layers=2,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through one message passing layer."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute messages for each edge."""
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        message = self.edge_mlp(edge_input)
        return message

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features after aggregation."""
        node_input = torch.cat([x, aggr_out], dim=-1)
        updated = self.node_mlp(node_input)
        return updated


class GNNSurrogate(nn.Module):
    """
    Complete Graph Neural Network for N-body dynamics.

    Architecture:
    1. Encode nodes and edges into embeddings
    2. Apply K message passing layers
    3. Decode node embeddings to accelerations
    """

    def __init__(
        self,
        node_input_dim: int = 7,
        edge_input_dim: int = 4,
        hidden_dim: int = 64,
        num_mp_layers: int = 3,
    ):
        """
        Args:
            node_input_dim: Raw node feature dimension (default: 7)
            edge_input_dim: Raw edge feature dimension (default: 4)
            hidden_dim: Hidden dimension for embeddings and MLPs
            num_mp_layers: Number of message passing layers
        """
        super().__init__()

        self.node_encoder = build_mlp(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )

        self.edge_encoder = build_mlp(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )

        self.mp_layers = nn.ModuleList(
            [
                InteractionLayer(
                    node_dim=hidden_dim, edge_dim=hidden_dim, hidden_dim=hidden_dim
                )
                for _ in range(num_mp_layers)
            ]
        )

        self.decoder = build_mlp(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=2,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            data: PyG Data object with:
                - x: Node features (N, node_input_dim)
                - edge_index: Edge connectivity (2, E)
                - edge_attr: Edge features (E, edge_input_dim)

        Returns:
            Predicted accelerations (N, 3)
        """
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for mp_layer in self.mp_layers:
            x = mp_layer(x, data.edge_index, edge_attr)

        accelerations = self.decoder(x)

        return accelerations
