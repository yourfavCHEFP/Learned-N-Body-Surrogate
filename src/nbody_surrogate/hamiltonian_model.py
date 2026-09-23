"""
src/nbody_surrogate/hamiltonian_model.py
Hamiltonian Neural Network (HNN) for N-body dynamics.
"""

import torch
import torch.nn as nn


class HamiltonianNN(nn.Module):
    """
    Learns H(q, p) such that:
        dq/dt =  dH/dp
        dp/dt = -dH/dq
    """

    def __init__(self, n_bodies: int = 3, hidden_dim: int = 128):
        super().__init__()
        # Input: positions (N*3) + velocities/momenta (N*3) + masses (N)
        in_dim = n_bodies * 3 * 2 + n_bodies
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # Scalar Hamiltonian (Energy)
        )

    def forward(self, q, p, m):
        # Concatenate into state vector
        bsz = q.shape[0]
        state = torch.cat([q.view(bsz, -1), p.view(bsz, -1), m.view(bsz, -1)], dim=-1)
        H = self.net(state)
        return H

    def time_derivative(self, q, p, m):
        """
        Computes (dq/dt, dp/dt) = (dH/dp, -dH/dq) using automatic differentiation.
        """
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = self.forward(q, p, m).sum()

        dH_dq = torch.autograd.grad(H, q, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H, p, create_graph=True)[0]

        dq_dt = dH_dp
        dp_dt = -dH_dq
        return dq_dt, dp_dt
