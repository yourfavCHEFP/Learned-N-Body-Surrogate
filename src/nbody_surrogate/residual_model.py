"""
src/nbody_surrogate/residual_model.py
GNN that predicts corrections (residuals) to analytical Newtonian gravity.
"""

import torch.nn as nn

from nbody_surrogate.model import GNNSurrogate

class ResidualGNNSurrogate(nn.Module):
    """
    Residual GNN:
        a_pred = a_analytical + scaling * GNN(state)
    """

    def __init__(
        self,
        node_input_dim: int = 7,
        edge_input_dim: int = 4,
        hidden_dim: int = 64,
        num_mp_layers: int = 3,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.gnn = GNNSurrogate(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_mp_layers=num_mp_layers,
        )
        self.residual_scale = residual_scale

    def forward(self, data):
        # Predict perturbation/residual
        delta_a = self.gnn(data)
        return delta_a
