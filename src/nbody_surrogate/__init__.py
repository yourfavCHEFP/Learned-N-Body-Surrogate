"""Learned N-Body Surrogate package."""

from .attention_model import AttentionGNNSurrogate
from .baseline import MLPBaseline
from .dataset import (
    Normalizer,
    compute_gravitational_accelerations,
    load_trajectory,
    trajectory_to_graphs,
)
from .hamiltonian_model import HamiltonianNN
from .model import GNNSurrogate, InteractionLayer
from .residual_model import ResidualGNNSurrogate
from .sim import milestone_system, run_simulation
from .symplectic_integrator import symplectic_rollout, velocity_verlet_step

__version__ = "0.1.0"
