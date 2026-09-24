"""
evaluate_stability.py
Compares MLP Baseline vs Standard GNN vs Residual GNN vs Rollout-Trained GNN
(and HNN, if trained) on real autoregressive rollout stability.

This used to just print a hand-typed table with no computation behind it at
all. It now actually rolls out every available checkpoint and measures:
  - Max Horizon (<1 AU): the first rollout step at which mean position error
    vs. the REBOUND ground truth exceeds 1 AU (or the full rollout length,
    if it never does).
  - Final Energy Drift (%): relative energy drift at the end of the rollout,
    using the trajectory's REAL G (see the G-constant fix in dataset.py/
    rollout.py) -- not the wrong-by-4*pi**2 constant this used to implicitly
    assume.

Any model whose checkpoint hasn't been trained yet is skipped with a clear
message rather than being given a fabricated number.
"""

from pathlib import Path

import numpy as np
import torch

from nbody_surrogate.baseline import MLPBaseline
from nbody_surrogate.dataset import Normalizer, load_trajectory
from nbody_surrogate.hamiltonian_model import HamiltonianNN
from nbody_surrogate.model import GNNSurrogate
from nbody_surrogate.residual_model import ResidualGNNSurrogate
from rollout import compute_energy, compute_trajectory_error, rollout_hnn, rollout_trajectory

DATA_PATH = Path("data/simulated/milestone_trajectory.npz")
ROLLOUT_STEPS = 500
DT = 0.01
STABILITY_THRESHOLD_AU = 1.0  # "stable" = mean position error under this


def _stable_horizon(pos_errors: np.ndarray) -> int:
    """First step index where error exceeds the threshold (or NaN/inf,
    i.e. the rollout diverged), else the full rollout length."""
    bad = ~np.isfinite(pos_errors) | (pos_errors > STABILITY_THRESHOLD_AU)
    exceeded = np.flatnonzero(bad)
    return int(exceeded[0]) if len(exceeded) else len(pos_errors) - 1


def _evaluate_gnn_family(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    initial_positions,
    initial_velocities,
    masses,
    true_positions,
    device,
    G: float,
    is_gnn: bool,
):
    checkpoint = checkpoint_dir / "best_model.pt"
    normalizer_path = checkpoint_dir / "normalizer.npz"
    if not checkpoint.exists() or not normalizer_path.exists():
        return None

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval().to(device)
    normalizer = Normalizer.load(normalizer_path)

    positions, velocities = rollout_trajectory(
        initial_positions, initial_velocities, masses, model,
        ROLLOUT_STEPS, DT, device, normalizer, G, is_gnn=is_gnn,
    )
    return _summarize(positions, velocities, true_positions, masses, G)


def _summarize(positions, velocities, true_positions, masses, G):
    pos_errors = compute_trajectory_error(positions, true_positions)
    horizon = _stable_horizon(pos_errors)

    energies = np.array(
        [compute_energy(positions[i], velocities[i], masses, G) for i in range(len(positions))]
    )
    if not np.all(np.isfinite(energies)) or energies[0] == 0:
        final_drift = float("inf")
    else:
        final_drift = 100.0 * abs(energies[-1] - energies[0]) / abs(energies[0])

    return {"horizon": horizon, "energy_drift_pct": final_drift}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trajectory = load_trajectory(str(DATA_PATH))
    initial_positions = trajectory["positions"][0]
    initial_velocities = trajectory["velocities"][0]
    masses = trajectory["masses"]
    G = trajectory["G"]
    n_bodies = len(masses)
    true_positions = trajectory["positions"][: ROLLOUT_STEPS + 1]

    print("=" * 60)
    print(f"Rollout Stability Evaluation ({ROLLOUT_STEPS} steps / {ROLLOUT_STEPS * DT:.1f} years)")
    print("=" * 60)

    results = {}

    results["MLP Baseline"] = _evaluate_gnn_family(
        MLPBaseline(input_dim=n_bodies * 7, hidden_dim=128, output_dim=n_bodies * 3, n_layers=2),
        Path("checkpoints/baseline"),
        initial_positions, initial_velocities, masses, true_positions, device, G, is_gnn=False,
    )

    results["Standard GNN"] = _evaluate_gnn_family(
        GNNSurrogate(node_input_dim=7, edge_input_dim=4, hidden_dim=64, num_mp_layers=3),
        Path("checkpoints/gnn"),
        initial_positions, initial_velocities, masses, true_positions, device, G, is_gnn=True,
    )

    results["Residual GNN"] = _evaluate_gnn_family(
        ResidualGNNSurrogate(node_input_dim=7, edge_input_dim=4, hidden_dim=64, num_mp_layers=3),
        Path("checkpoints/residual_gnn"),
        initial_positions, initial_velocities, masses, true_positions, device, G, is_gnn=True,
    )

    results["Rollout-Trained GNN"] = _evaluate_gnn_family(
        GNNSurrogate(node_input_dim=7, edge_input_dim=4, hidden_dim=64, num_mp_layers=3),
        Path("checkpoints/rollout_gnn"),
        initial_positions, initial_velocities, masses, true_positions, device, G, is_gnn=True,
    )

    hnn_checkpoint = Path("checkpoints/hnn/best_model.pt")
    if hnn_checkpoint.exists():
        hnn_model = HamiltonianNN(n_bodies=n_bodies, hidden_dim=128).to(device)
        hnn_model.load_state_dict(torch.load(hnn_checkpoint, map_location=device, weights_only=True))
        positions, velocities = rollout_hnn(
            initial_positions, initial_velocities, masses, hnn_model, ROLLOUT_STEPS, DT, device
        )
        results["Hamiltonian NN"] = _summarize(positions, velocities, true_positions, masses, G)
    else:
        results["Hamiltonian NN"] = None

    print(f"{'Model':<22} | {'Max Horizon (<1 AU)':<22} | {'Final Energy Drift':<20}")
    print("-" * 68)
    for name, r in results.items():
        if r is None:
            print(f"{name:<22} | {'checkpoint not found':<22} | {'-':<20}")
        else:
            drift_str = "diverged" if not np.isfinite(r["energy_drift_pct"]) else f"{r['energy_drift_pct']:.1f} %"
            print(f"{name:<22} | {r['horizon']:<4} / {ROLLOUT_STEPS:<15} | {drift_str:<20}")
    print("=" * 68)


if __name__ == "__main__":
    main()
