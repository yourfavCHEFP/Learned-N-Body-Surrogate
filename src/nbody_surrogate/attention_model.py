"""
src/nbody_surrogate/attention_model.py

Graph Attention Network (GAT) surrogate for gravitational N-body dynamics.
Computes learnable edge attention weights representing pairwise interaction strengths.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class AttentionInteractionLayer(MessagePassing):
    """
    Message-passing layer with multi-head or single-head edge attention.
    Learns to dynamically weight gravitational influences between bodies.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="add", flow="source_to_target")

        # Edge feature transformation & message network
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention score computation network
        self.attn_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Node update network
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.last_attention_weights = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [N, node_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        updated_x = self.update_mlp(torch.cat([x, out], dim=-1))
        return updated_x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        # Concatenate node states and edge attributes: [E, 2*node_dim + edge_dim]
        pair_feat = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute unnormalized attention scores and normalize across incoming edges
        alpha = self.attn_mlp(pair_feat)
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)
        self.last_attention_weights = alpha.detach()

        msg = self.msg_mlp(pair_feat)
        return alpha * msg

class AttentionGNNSurrogate(nn.Module):
    """
    Complete Graph Attention Network Surrogate for predicting accelerations.
    """

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_dim: int = 3,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, edge_in_dim),
        )

        self.layers = nn.ModuleList(
            [
                AttentionInteractionLayer(
                    node_dim=hidden_dim, edge_dim=edge_in_dim, hidden_dim=hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h = layer(h, edge_index, e)

        accel = self.decoder(h)
        return accel

    def get_attention_weights(self, data) -> torch.Tensor:
        """
        Runs a forward pass and returns the attention weights from the first layer.
        """
        _ = self.forward(data)
        return self.layers[0].last_attention_weights.squeeze(-1)
