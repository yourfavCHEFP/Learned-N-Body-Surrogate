"""Load simulated trajectories and convert them into graph samples.

Two bugs fixed here (see project review):

1. G-constant bug: `compute_gravitational_accelerations` used to default to
   G=1.0, but REBOUND (units yr/AU/Msun) actually integrates with
   G=4*pi**2 (~39.48). Every acceleration "ground truth" computed with the
   old default was ~39.5x too weak relative to the dynamics that actually
   produced the trajectory. G is now REQUIRED (no silent wrong default) and
   is read from the trajectory itself (`sim.py` now stores it).

2. Normalizer was fit but never applied. `trajectory_to_graphs` now accepts
   an optional `normalizer` and, when given one, actually normalizes node
   features, edge features, and targets before building the graphs.

Import layering: torch / torch_geometric are imported lazily so that a
script which only needs `load_trajectory` / `compute_gravitational_accelerations`
(pure REBOUND/numpy work) never has to have PyTorch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

try:
    import torch
    from torch_geometric.data import Data

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in torch-free envs
    torch = None  # type: ignore[assignment]
    Data = None  # type: ignore[assignment,misc]
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "This function builds PyTorch Geometric graphs and needs torch + "
            "torch_geometric installed. `load_trajectory` and "
            "`compute_gravitational_accelerations` work without them."
        )


# G for REBOUND's ("yr", "AU", "Msun") unit convention -- used only as a
# fallback for OLD trajectory files saved before SimResult stored its own G.
_DEFAULT_YR_AU_MSUN_G = 4.0 * np.pi**2

_warned_missing_G = False


class Trajectory(TypedDict):
    """Arrays stored by ``SimResult.save``."""

    times: np.ndarray  # Shape: (T,)
    positions: np.ndarray  # Shape: (T, N, 3)
    velocities: np.ndarray  # Shape: (T, N, 3)
    masses: np.ndarray  # Shape: (N,)
    G: float  # the gravitational constant this trajectory was generated with


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

    def normalize_edge_features(self, edge_features: np.ndarray) -> np.ndarray:
        """
        edge_features columns are [dx, dy, dz, distance] (see
        `trajectory_to_graphs`). Relative-position vectors are differences of
        two positions, so they're already centered near zero by construction
        -- we scale (don't re-center) them by the position spread. Distance
        is scaled by the mean per-axis position spread.
        """
        rel_pos = edge_features[..., :3] / self.pos_std
        dist_scale = float(np.mean(self.pos_std))
        dist = edge_features[..., 3:4] / dist_scale
        return np.concatenate([rel_pos, dist], axis=-1)

    def save(self, path: str | Path) -> None:
        """Persist these stats so eval/rollout scripts don't have to refit
        (and risk silently drifting from the exact stats a checkpoint was
        trained with)."""
        np.savez(
            path,
            mass_mean=self.mass_mean,
            mass_std=self.mass_std,
            pos_mean=self.pos_mean,
            pos_std=self.pos_std,
            vel_mean=self.vel_mean,
            vel_std=self.vel_std,
            acc_mean=self.acc_mean,
            acc_std=self.acc_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        with np.load(path) as d:
            return cls(
                mass_mean=float(d["mass_mean"]),
                mass_std=float(d["mass_std"]),
                pos_mean=d["pos_mean"],
                pos_std=d["pos_std"],
                vel_mean=d["vel_mean"],
                vel_std=d["vel_std"],
                acc_mean=d["acc_mean"],
                acc_std=d["acc_std"],
            )

    def denormalize_acc(
        self, norm_acc: "torch.Tensor | np.ndarray",
    ) -> "torch.Tensor | np.ndarray":
        """Map a model's normalized-acceleration output back to physical units.
        ALWAYS call this on model output before using it for physics
        integration (e.g. in an autoregressive rollout) -- the model's raw
        output lives in normalized space, not real acceleration units."""
        if _HAS_TORCH and isinstance(norm_acc, torch.Tensor):
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
    global _warned_missing_G
    with np.load(path) as archive:
        required = {"times", "positions", "velocities", "masses"}
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Trajectory is missing required arrays: {names}")

        if "G" in archive.files:
            G = float(archive["G"])
        else:
            G = _DEFAULT_YR_AU_MSUN_G
            if not _warned_missing_G:
                print(
                    f"WARNING: {path} has no stored G (saved before this fix). "
                    f"Falling back to G={G:.6f} (REBOUND's yr/AU/Msun value). "
                    "If this trajectory used different units, this is WRONG -- "
                    "regenerate it with the current sim.py."
                )
                _warned_missing_G = True

        trajectory: Trajectory = {
            "times": np.asarray(archive["times"]),
            "positions": np.asarray(archive["positions"]),
            "velocities": np.asarray(archive["velocities"]),
            "masses": np.asarray(archive["masses"]),
            "G": G,
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
    positions: np.ndarray,
    masses: np.ndarray,
    G: float,
    softening: float = 1e-6,
) -> np.ndarray:
    """
    Compute Newtonian gravitational accelerations for all bodies across all snapshots.

    Args:
        positions: Array of shape (T, N, 3)
        masses: Array of shape (N,)
        G: Gravitational constant. REQUIRED, no default -- pass trajectory["G"].
           Do NOT assume 1.0: REBOUND's ("yr","AU","Msun") units give
           G=4*pi**2 (~39.48), not 1. Using the wrong G silently produces
           acceleration targets that don't match the trajectory's real dynamics.
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


def _complete_graph_edges(n_bodies: int):
    """Return directed edges between every pair of distinct bodies."""
    _require_torch()
    source, target = np.meshgrid(
        np.arange(n_bodies), np.arange(n_bodies), indexing="ij"
    )
    mask = source != target
    return torch.from_numpy(np.stack((source[mask], target[mask]))).long()


def trajectory_to_graphs(
    trajectory: Trajectory, normalizer: "Normalizer | None" = None
) -> "list[Data]":
    """
    Convert consecutive trajectory snapshots into one-step graph samples.

    Args:
        trajectory: as returned by `load_trajectory` (must include "G").
        normalizer: if given, node features, edge features, and targets are
            all normalized through it before being packed into `Data`
            objects -- this is what actually makes the Normalizer's stats
            reach the model. Fit it ONCE on your training trajectory and
            reuse the SAME instance everywhere else (eval, rollout) --
            never refit on a different/test trajectory, or you silently
            evaluate the model on a different feature scale than it was
            trained on.
            If None, graphs are built with raw (unnormalized) values, e.g.
            for quick physics inspection.
    """
    _require_torch()
    positions = trajectory["positions"]
    velocities = trajectory["velocities"]
    masses = trajectory["masses"]
    times = trajectory["times"]
    G = trajectory["G"]
    n_snapshots, _, _ = positions.shape

    if n_snapshots < 2:
        raise ValueError("At least two snapshots are required to create targets")

    # Compute accelerations as targets, using THIS trajectory's own G.
    accelerations = compute_gravitational_accelerations(positions, masses, G=G)

    edge_index = _complete_graph_edges(positions.shape[1])
    source = edge_index[0].numpy()
    target = edge_index[1].numpy()
    graphs: list[Data] = []

    norm_masses = normalizer.normalize_mass(masses) if normalizer else masses

    for index in range(n_snapshots - 1):
        pos_t = positions[index]
        vel_t = velocities[index]
        targets = accelerations[index]

        if normalizer is not None:
            pos_feat = normalizer.normalize_pos(pos_t)
            vel_feat = normalizer.normalize_vel(vel_t)
            targets = normalizer.normalize_acc(targets)
        else:
            pos_feat = pos_t
            vel_feat = vel_t

        # Node features: [mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        state = np.concatenate(
            (norm_masses[:, None], pos_feat, vel_feat), axis=1
        )

        # Edge features: relative position vectors (RAW positions -- edge
        # geometry should reflect true relative displacement, then get its
        # own scaling below).
        relative_positions = pos_t[target] - pos_t[source]
        distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)
        edge_features = np.concatenate([relative_positions, distances], axis=1)
        if normalizer is not None:
            edge_features = normalizer.normalize_edge_features(edge_features)

        graphs.append(
            Data(
                x=torch.from_numpy(state).float(),
                y=torch.from_numpy(targets).float(),
                # a_prior: the analytical Newtonian acceleration, used by
                # ResidualGNNSurrogate as `a_pred = a_prior + scale*GNN(state)`.
                # NOTE: for this dataset a_prior == y exactly, because REBOUND's
                # ground truth here IS pure point-mass Newtonian gravity with
                # no un-modeled physics -- so a correctly-trained residual
                # model should learn scale*GNN(state) ~= 0. That's expected,
                # not a bug: the architecture becomes non-trivial the moment
                # the ground truth includes something this analytical formula
                # doesn't capture (e.g. a REBOUNDx force, GR corrections, or
                # real ephemeris data with real observational structure).
                a_prior=torch.from_numpy(targets).float(),
                edge_index=edge_index,
                edge_attr=torch.from_numpy(edge_features).float(),
                time=torch.tensor(times[index], dtype=torch.float32),
                delta_t=torch.tensor(
                    times[index + 1] - times[index], dtype=torch.float32
                ),
            )
        )

    return graphs
