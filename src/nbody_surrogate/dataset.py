"""Load simulated trajectories and convert them into graph samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch_geometric.data import Data


class Trajectory(TypedDict):
    """Arrays stored by ``SimResult.save``."""

    times: np.ndarray  # Shape: (T,)
    positions: np.ndarray  # Shape: (T, N, 3)
    velocities: np.ndarray  # Shape: (T, N, 3)
    masses: np.ndarray  # Shape: (N,)


@dataclass
class Normalizer:
    """Stores normalization statistics and provides forward/inverse transforms."""

    mass_mean: float
    mass_std: float
    pos_mean: np.ndarray  # (3,)
    pos_std: np.ndarray  # (3,)
    vel_mean: np.ndarray  # (3,)
    vel_std: np.ndarray  # (3,)
    acc_mean: np.ndarray  # (3,)
    acc_std: np.ndarray  # (3,)

    @classmethod
    def fit(
        cls,
        masses: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> Normalizer:
        """Compute normalization statistics from training trajectory arrays."""
        eps = 1e-8
        log_masses = np.log10(masses + eps)

        return cls(
            mass_mean=float(np.mean(log_masses)),
            mass_std=float(np.std(log_masses) + eps),
            pos_mean=np.mean(positions, axis=(0, 1)),
            pos_std=np.std(positions, axis=(0, 1)) + eps,
            vel_mean=np.mean(velocities, axis=(0, 1)),
            vel_std=np.std(velocities, axis=(0, 1)) + eps,
            acc_mean=np.mean(accelerations, axis=(0, 1)),
            acc_std=np.std(accelerations, axis=(0, 1)) + eps,
        )

    def normalize_mass(self, masses: np.ndarray) -> np.ndarray:
        eps = 1e-8
        log_m = np.log10(masses + eps)
        return (log_m - self.mass_mean) / self.mass_std

    def normalize_pos(self, positions: np.ndarray) -> np.ndarray:
        return (positions - self.pos_mean) / self.pos_std

    def normalize_vel(self, velocities: np.ndarray) -> np.ndarray:
        return (velocities - self.vel_mean) / self.vel_std

    def normalize_acc(self, accelerations: np.ndarray) -> np.ndarray:
        return (accelerations - self.acc_mean) / self.acc_std

    def denormalize_acc(
        self, norm_acc: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        if isinstance(norm_acc, torch.Tensor):
            mean = torch.as_tensor(
                self.acc_mean, dtype=norm_acc.dtype, device=norm_acc.device
            )
            std = torch.as_tensor(
                self.acc_std, dtype=norm_acc.dtype, device=norm_acc.device
            )
            return norm_acc * std + mean
        return norm_acc * self.acc_std + self.acc_mean


def load_trajectory(path: str | Path) -> Trajectory:
    """Load and validate a trajectory saved with ``SimResult.save``."""
    with np.load(path) as archive:
        required = {"times", "positions", "velocities", "masses"}
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Trajectory is missing required arrays: {names}")

        trajectory: Trajectory = {
            "times": np.asarray(archive["times"]),
            "positions": np.asarray(archive["positions"]),
            "velocities": np.asarray(archive["velocities"]),
            "masses": np.asarray(archive["masses"]),
        }

    times = trajectory["times"]
    positions = trajectory["positions"]
    velocities = trajectory["velocities"]
    masses = trajectory["masses"]

    if times.ndim != 1:
        raise ValueError("times must have shape (T,)")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (T, N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions")
    if masses.shape != (positions.shape[1],):
        raise ValueError("masses must have shape (N,)")
    if len(times) != positions.shape[0]:
        raise ValueError(
            "times and positions must contain the same number of snapshots"
        )

    return trajectory


def compute_gravitational_accelerations(
    positions: np.ndarray, masses: np.ndarray, G: float = 1.0, softening: float = 1e-6
) -> np.ndarray:
    """
    Compute Newtonian gravitational accelerations for all bodies across all snapshots.

    Args:
        positions: Array of shape (T, N, 3)
        masses: Array of shape (N,)
        G: Gravitational constant (default G=1.0 for REBOUND solar units)
        softening: Small value to prevent division by zero

    Returns:
        accelerations: Array of shape (T, N, 3)
    """
    T, N, _ = positions.shape
    accelerations = np.zeros_like(positions)

    for t in range(T):
        for i in range(N):
            acc = np.zeros(3)
            for j in range(N):
                if i != j:
                    r_vec = positions[t, j] - positions[t, i]
                    r_mag = np.sqrt(np.sum(r_vec**2) + softening**2)
                    acc += G * masses[j] * r_vec / r_mag**3
            accelerations[t, i] = acc

    return accelerations


def _complete_graph_edges(n_bodies: int) -> torch.Tensor:
    """Return directed edges between every pair of distinct bodies."""
    source, target = np.meshgrid(
        np.arange(n_bodies), np.arange(n_bodies), indexing="ij"
    )
    mask = source != target
    return torch.from_numpy(np.stack((source[mask], target[mask]))).long()


def trajectory_to_graphs(trajectory: Trajectory) -> list[Data]:
    """Convert consecutive trajectory snapshots into one-step graph samples."""
    positions = trajectory["positions"]
    velocities = trajectory["velocities"]
    masses = trajectory["masses"]
    times = trajectory["times"]
    n_snapshots, _, _ = positions.shape

    if n_snapshots < 2:
        raise ValueError("At least two snapshots are required to create targets")

    # Compute accelerations as targets
    accelerations = compute_gravitational_accelerations(positions, masses)

    edge_index = _complete_graph_edges(positions.shape[1])
    source = edge_index[0].numpy()
    target = edge_index[1].numpy()
    graphs: list[Data] = []

    for index in range(n_snapshots - 1):
        # Node features: [mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        state = np.concatenate(
            (masses[:, None], positions[index], velocities[index]), axis=1
        )

        # Targets: accelerations at current timestep
        targets = accelerations[index]

        # Edge features: relative position vectors
        relative_positions = positions[index][target] - positions[index][source]
        distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)
        edge_features = np.concatenate([relative_positions, distances], axis=1)


        graphs.append(
            Data(
                x=torch.from_numpy(state).float(),
                y=torch.from_numpy(targets).float(),
                edge_index=edge_index,
                edge_attr=torch.from_numpy(edge_features).float(),
                time=torch.tensor(times[index], dtype=torch.float32),
                delta_t=torch.tensor(
                    times[index + 1] - times[index], dtype=torch.float32
                ),
            )
        )

    return graphs
