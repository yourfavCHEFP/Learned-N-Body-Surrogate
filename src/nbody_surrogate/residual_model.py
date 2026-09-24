"""
src/nbody_surrogate/residual_model.py
GNN that predicts corrections (residuals) to analytical Newtonian gravity.
"""

import torch.nn as nn

from nbody_surrogate.model import GNNSurrogate


class ResidualGNNSurrogate(nn.Module):
    """
    Residual GNN:
        a_pred = a_prior + residual_scale * GNN(state)

    where `a_prior` is the analytical Newtonian acceleration for the current
    state (see `dataset.trajectory_to_graphs`, which attaches it to each
    graph as `data.a_prior`, in the SAME normalized space as `data.y`).

    IMPORTANT CAVEAT (read before trusting a near-zero residual as a sign of
    a working model): on the current synthetic milestone dataset, `a_prior`
    is computed with the exact same Newtonian formula used to generate
    REBOUND's ground truth, so `a_prior == data.y` up to floating point and
    softening (negligible at AU-scale separations). A correctly-trained
    residual model on THIS dataset should therefore converge to
    `GNN(state) ~= 0` everywhere -- that's the architecture doing its job
    correctly, not evidence the model is "better." The residual only becomes
    a meaningful learning target once the ground truth contains something
    the analytical formula can't express: e.g. a REBOUNDx force module
    (`rebx_evolve`, general-relativistic corrections), or real ephemeris data.
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
        if not hasattr(data, "a_prior"):
            raise AttributeError(
                "ResidualGNNSurrogate requires `data.a_prior` (the analytical "
                "Newtonian acceleration for this state). Build graphs with "
                "`trajectory_to_graphs`, which attaches it automatically."
            )
        delta_a = self.gnn(data)
        return data.a_prior + self.residual_scale * delta_a
