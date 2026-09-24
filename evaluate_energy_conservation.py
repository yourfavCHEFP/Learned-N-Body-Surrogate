"""
evaluate_energy_conservation.py
Compares energy conservation across REBOUND ground truth, the Standard GNN,
and the Hamiltonian NN -- via real autoregressive rollouts.

Previously the GNN and HNN numbers here ("~193%", "<10%") were hardcoded
strings and hardcoded plot reference lines, not computed from any model
rollout. This version actually loads each checkpoint, rolls it out, and
computes its own energy-drift curve, using the trajectory's REAL G
(REBOUND's yr/AU/Msun convention gives G=4*pi**2, not 1 -- see the
G-constant fix elsewhere in this project).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nbody_surrogate.dataset import Normalizer, load_trajectory
from nbody_surrogate.hamiltonian_model import HamiltonianNN
from nbody_surrogate.model import GNNSurrogate
from rollout import compute_energy, rollout_hnn, rollout_trajectory

ROLLOUT_STEPS = 500
DT = 0.01


def relative_drift(energies: np.ndarray) -> np.ndarray:
    return 100 * (energies - energies[0]) / abs(energies[0])


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Energy Conservation Comparison")
    print("=" * 60)

    trajectory = load_trajectory("data/simulated/milestone_trajectory.npz")
    masses = trajectory["masses"]
    G = trajectory["G"]  # REAL constant, not an assumed 1.0
    n_bodies = len(masses)

    initial_positions = trajectory["positions"][0]
    initial_velocities = trajectory["velocities"][0]

    rebound_pos = trajectory["positions"][: ROLLOUT_STEPS + 1]
    rebound_vel = trajectory["velocities"][: ROLLOUT_STEPS + 1]
    rebound_times = trajectory["times"][: ROLLOUT_STEPS + 1]

    rebound_energy = np.array(
        [compute_energy(rebound_pos[i], rebound_vel[i], masses, G) for i in range(len(rebound_pos))]
    )
    rebound_drift_pct = 100 * abs(rebound_energy[-1] - rebound_energy[0]) / abs(rebound_energy[0])
    print(f"\nREBOUND (Ground Truth):\n  Energy drift over {len(rebound_times)} steps: {rebound_drift_pct:.6f}%")
    print("  (should be ~machine precision -- WHFast is symplectic; a large")
    print("   number here would mean G is wrong again, not real drift)")

    curves = {"REBOUND": relative_drift(rebound_energy)}
    final_drifts = {"REBOUND": rebound_drift_pct}

    gnn_checkpoint = Path("checkpoints/gnn/best_model.pt")
    gnn_normalizer_path = Path("checkpoints/gnn/normalizer.npz")
    if gnn_checkpoint.exists() and gnn_normalizer_path.exists():
        normalizer = Normalizer.load(gnn_normalizer_path)
        model = GNNSurrogate(node_input_dim=7, edge_input_dim=4, hidden_dim=64, num_mp_layers=3).to(device)
        model.load_state_dict(torch.load(gnn_checkpoint, map_location=device, weights_only=True))
        model.eval()

        positions, velocities = rollout_trajectory(
            initial_positions, initial_velocities, masses, model,
            ROLLOUT_STEPS, DT, device, normalizer, G, is_gnn=True,
        )
        gnn_energy = np.array(
            [compute_energy(positions[i], velocities[i], masses, G) for i in range(len(positions))]
        )
        gnn_drift_pct = 100 * abs(gnn_energy[-1] - gnn_energy[0]) / abs(gnn_energy[0])
        curves["Standard GNN"] = relative_drift(gnn_energy)
        final_drifts["Standard GNN"] = gnn_drift_pct
        print(f"\nStandard GNN:\n  Energy drift: {gnn_drift_pct:.2f}% (measured over a real {ROLLOUT_STEPS}-step rollout)")
    else:
        print("\n⚠️  Standard GNN checkpoint not found")

    hnn_checkpoint = Path("checkpoints/hnn/best_model.pt")
    if hnn_checkpoint.exists():
        model = HamiltonianNN(n_bodies=n_bodies, hidden_dim=128).to(device)
        model.load_state_dict(torch.load(hnn_checkpoint, map_location=device, weights_only=True))

        positions, velocities = rollout_hnn(
            initial_positions, initial_velocities, masses, model, ROLLOUT_STEPS, DT, device
        )
        hnn_energy = np.array(
            [compute_energy(positions[i], velocities[i], masses, G) for i in range(len(positions))]
        )
        hnn_drift_pct = 100 * abs(hnn_energy[-1] - hnn_energy[0]) / abs(hnn_energy[0])
        curves["Hamiltonian NN"] = relative_drift(hnn_energy)
        final_drifts["Hamiltonian NN"] = hnn_drift_pct
        if "Standard GNN" in final_drifts and final_drifts["Standard GNN"] > 0:
            improvement = final_drifts["Standard GNN"] / max(hnn_drift_pct, 1e-9)
            print(f"\nHamiltonian NN:\n  Energy drift: {hnn_drift_pct:.2f}%")
            print(f"  Improvement over Standard GNN: {improvement:.1f}x")
        else:
            print(f"\nHamiltonian NN:\n  Energy drift: {hnn_drift_pct:.2f}%")
    else:
        print("\n⚠️  HNN checkpoint not found - run train_hnn.py first")

    # --- Plot: real curves only, no hardcoded reference lines ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"REBOUND": "black", "Standard GNN": "tab:red", "Hamiltonian NN": "tab:green"}

    for name, curve in curves.items():
        style = "--" if name != "REBOUND" else "-"
        ax1.plot(rebound_times, curve, style, color=colors.get(name), linewidth=2, alpha=0.8, label=name)
    ax1.set_xlabel("Time (years)", fontsize=12)
    ax1.set_ylabel("Relative Energy Drift (%)", fontsize=12)
    ax1.set_title("Energy Drift Over Time (linear)", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, curve in curves.items():
        style = "--" if name != "REBOUND" else "-"
        ax2.plot(rebound_times, np.abs(curve) + 1e-12, style, color=colors.get(name), linewidth=2, alpha=0.8,
                  label=f"{name} ({final_drifts[name]:.2f}%)")
    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("|Relative Energy Drift| (%)", fontsize=12)
    ax2.set_yscale("log")
    ax2.set_title("Energy Drift Over Time (log)", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "energy_conservation_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved comparison plot to {save_path}")

    print("\n" + "=" * 60)
    print("Summary (all measured, not hardcoded):")
    for name, drift in final_drifts.items():
        print(f"  {name:<16}: {drift:.4f}% drift")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
