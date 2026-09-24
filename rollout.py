"""Autoregressive rollout: Multi-step trajectory integration using trained surrogates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from nbody_surrogate.baseline import MLPBaseline
from nbody_surrogate.dataset import (
    Normalizer,
    compute_gravitational_accelerations,
    load_trajectory,
    trajectory_to_graphs,
)
from nbody_surrogate.model import GNNSurrogate

# Paths
DATA_PATH = Path("data/simulated/milestone_trajectory.npz")
MLP_CHECKPOINT = Path("checkpoints/baseline/best_model.pt")
GNN_CHECKPOINT = Path("checkpoints/gnn/best_model.pt")
OUTPUT_DIR = Path("checkpoints")

# Rollout parameters
ROLLOUT_STEPS = 500
DT = 0.01
# NOTE: G is no longer a hardcoded constant here. It used to default to 1.0,
# but REBOUND (units yr/AU/Msun) actually integrates with G=4*pi**2 (~39.48).
# Using the wrong G here doesn't just mis-scale energy -- it means energy
# computed even for the REAL REBOUND trajectory drifts, because
# (KE - G_wrong * PE_shape) isn't the system's actual conserved quantity.
# `main()` now reads the real G from the loaded trajectory and passes it
# through explicitly everywhere energy is computed.


def _build_gnn_graph(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    normalizer: Normalizer,
    device: torch.device,
    G: float,
) -> Data:
    """Build one normalized graph for the GNN, matching trajectory_to_graphs'
    feature construction exactly (raw edge geometry, then normalized).

    Also attaches `a_prior` (the analytical Newtonian acceleration at this
    state, normalized) so this same builder works for both GNNSurrogate and
    ResidualGNNSurrogate -- the latter reads data.a_prior internally, the
    former simply ignores the extra field.
    """
    n_bodies = len(masses)

    node_features = np.concatenate(
        [
            normalizer.normalize_mass(masses)[:, None],
            normalizer.normalize_pos(positions),
            normalizer.normalize_vel(velocities),
        ],
        axis=1,
    )

    edge_index = []
    edge_attr = []
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                edge_index.append([i, j])
                rel_pos = positions[j] - positions[i]
                distance = np.linalg.norm(rel_pos)
                edge_attr.append(np.concatenate([rel_pos, [distance]]))
    edge_attr = normalizer.normalize_edge_features(np.array(edge_attr))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    a_prior = compute_gravitational_accelerations(
        positions[None, ...], masses, G=G
    )[0]
    a_prior = normalizer.normalize_acc(a_prior)

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        a_prior=torch.tensor(a_prior, dtype=torch.float32),
    ).to(device)


def _build_mlp_input(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    normalizer: Normalizer,
    device: torch.device,
) -> torch.Tensor:
    """Build one normalized flattened-state input for the MLP baseline."""
    flat_input = np.concatenate(
        [
            normalizer.normalize_mass(masses)[:, None],
            normalizer.normalize_pos(positions),
            normalizer.normalize_vel(velocities),
        ],
        axis=1,
    ).flatten()
    return torch.tensor(flat_input, dtype=torch.float32).unsqueeze(0).to(device)


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    masses: np.ndarray,
    model: torch.nn.Module,
    dt: float,
    device: torch.device,
    normalizer: Normalizer,
    G: float,
    is_gnn: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one Velocity Verlet integration step.

    `accelerations` in and out are always in REAL physical units -- the model
    is only ever asked for a normalized-space prediction internally, which is
    immediately denormalized via `normalizer.denormalize_acc` before it's
    used in the physics update below. Never feed a model's raw output
    directly into position/velocity updates.
    """
    n_bodies = len(masses)

    new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2

    if is_gnn:
        graph = _build_gnn_graph(new_positions, velocities, masses, normalizer, device, G)
        with torch.no_grad():
            new_accelerations = model(graph).cpu().numpy()
    else:
        flat_input = _build_mlp_input(new_positions, velocities, masses, normalizer, device)
        with torch.no_grad():
            flat_output = model(flat_input).cpu().numpy()
        new_accelerations = flat_output.reshape(n_bodies, 3)

    new_accelerations = normalizer.denormalize_acc(new_accelerations)

    new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt

    return new_positions, new_velocities, new_accelerations


def rollout_trajectory(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    masses: np.ndarray,
    model: torch.nn.Module,
    num_steps: int,
    dt: float,
    device: torch.device,
    normalizer: Normalizer,
    G: float,
    is_gnn: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run autoregressive rollout for num_steps timesteps."""
    n_bodies = len(masses)

    positions_history = np.zeros((num_steps + 1, n_bodies, 3))
    velocities_history = np.zeros((num_steps + 1, n_bodies, 3))

    positions_history[0] = initial_positions
    velocities_history[0] = initial_velocities

    if is_gnn:
        graph = _build_gnn_graph(
            initial_positions, initial_velocities, masses, normalizer, device, G
        )
        with torch.no_grad():
            accelerations = model(graph).cpu().numpy()
    else:
        flat_input = _build_mlp_input(
            initial_positions, initial_velocities, masses, normalizer, device
        )
        with torch.no_grad():
            flat_output = model(flat_input).cpu().numpy()
        accelerations = flat_output.reshape(n_bodies, 3)

    accelerations = normalizer.denormalize_acc(accelerations)

    positions = initial_positions.copy()
    velocities = initial_velocities.copy()

    for step in range(num_steps):
        positions, velocities, accelerations = velocity_verlet_step(
            positions,
            velocities,
            accelerations,
            masses,
            model,
            dt,
            device,
            normalizer,
            G,
            is_gnn,
        )

        positions_history[step + 1] = positions
        velocities_history[step + 1] = velocities

    return positions_history, velocities_history


def rollout_hnn(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    masses: np.ndarray,
    model: torch.nn.Module,
    num_steps: int,
    dt: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Autoregressive rollout for HamiltonianNN.

    HNN predicts (dq/dt, dp/dt) = (dH/dp, -dH/dq) directly via autograd, not
    an acceleration to feed into velocity-Verlet -- so it needs its own
    integrator, not `rollout_trajectory`/`velocity_verlet_step` above.
    Uses plain explicit Euler on (q, p), matching the same finite-difference
    convention `train_hnn.py` used to build its training targets.
    """
    n_bodies = len(masses)
    positions_history = np.zeros((num_steps + 1, n_bodies, 3))
    velocities_history = np.zeros((num_steps + 1, n_bodies, 3))

    positions_history[0] = initial_positions
    velocities_history[0] = initial_velocities

    q = torch.tensor(initial_positions, dtype=torch.float32, device=device).unsqueeze(0)
    p = torch.tensor(
        initial_velocities * masses[:, None], dtype=torch.float32, device=device
    ).unsqueeze(0)
    m = torch.tensor(masses, dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    for step in range(num_steps):
        # time_derivative needs gradient tracking internally (see train_hnn.py);
        # we still don't want it to build a graph across steps, so detach
        # immediately after each step's derivative is computed.
        dq_dt, dp_dt = model.time_derivative(q, p, m)
        q = (q + dt * dq_dt).detach()
        p = (p + dt * dp_dt).detach()

        positions_history[step + 1] = q.squeeze(0).cpu().numpy()
        velocities_history[step + 1] = (p.squeeze(0).cpu().numpy()) / masses[:, None]

    return positions_history, velocities_history


def compute_energy(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, G: float
) -> float:
    """Compute total energy (kinetic + potential). G is REQUIRED -- pass the
    trajectory's own G (see dataset.py); do not assume 1.0."""
    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    potential = 0.0
    n_bodies = len(masses)
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            r_ij = np.linalg.norm(positions[j] - positions[i])
            potential -= G * masses[i] * masses[j] / r_ij

    return kinetic + potential


def compute_trajectory_error(
    pred_positions: np.ndarray, true_positions: np.ndarray
) -> np.ndarray:
    """Compute mean position error at each timestep."""
    errors = np.linalg.norm(pred_positions - true_positions, axis=2)
    mean_error = np.mean(errors, axis=1)
    return mean_error


def plot_rollout_comparison(
    times: np.ndarray,
    true_positions: np.ndarray,
    gnn_positions: np.ndarray,
    mlp_positions: np.ndarray,
    gnn_energies: np.ndarray,
    mlp_energies: np.ndarray,
    true_energies: np.ndarray,
    output_path: Path,
) -> None:
    """Generate comprehensive 6-panel rollout diagnostic plot."""
    fig = plt.figure(figsize=(18, 12))

    body_names = ["Star", "Planet 1 (Inner)", "Planet 2 (Outer)"]
    colors = ["gold", "dodgerblue", "orangered"]

    # Panel 1: XY Orbital Trajectories (GNN)
    ax1 = plt.subplot(2, 3, 1)
    for i, (name, color) in enumerate(zip(body_names, colors)):
        ax1.plot(
            true_positions[:, i, 0],
            true_positions[:, i, 1],
            "-",
            color=color,
            linewidth=2,
            label=f"{name} (REBOUND)",
            alpha=0.8,
        )
        ax1.plot(
            gnn_positions[:, i, 0],
            gnn_positions[:, i, 1],
            "--",
            color=color,
            linewidth=1.5,
            label=f"{name} (GNN)",
            alpha=0.6,
        )

    ax1.scatter(
        [0],
        [0],
        s=300,
        c="gold",
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    ax1.set_xlabel("x (AU)", fontsize=11)
    ax1.set_ylabel("y (AU)", fontsize=11)
    ax1.set_title(
        "Orbital Trajectories (XY): GNN vs. Ground Truth",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Panel 2: Position Error Over Time
    ax2 = plt.subplot(2, 3, 2)
    gnn_errors = compute_trajectory_error(gnn_positions, true_positions)
    mlp_errors = compute_trajectory_error(mlp_positions, true_positions)

    ax2.plot(times, gnn_errors, linewidth=2, label="GNN Surrogate", color="tab:blue")
    ax2.plot(times, mlp_errors, linewidth=2, label="MLP Baseline", color="tab:orange")
    ax2.set_xlabel("Time (years)", fontsize=11)
    ax2.set_ylabel("Mean Position Error (AU)", fontsize=11)
    ax2.set_title("Position Error vs. Time", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Panel 3: Energy Conservation
    ax3 = plt.subplot(2, 3, 3)
    E0_true = true_energies[0]

    gnn_energy_drift = (
        np.abs(gnn_energies - gnn_energies[0]) / np.abs(gnn_energies[0]) * 100
    )
    mlp_energy_drift = (
        np.abs(mlp_energies - mlp_energies[0]) / np.abs(mlp_energies[0]) * 100
    )
    true_energy_drift = np.abs(true_energies - E0_true) / np.abs(E0_true) * 100

    ax3.plot(times, gnn_energy_drift, linewidth=2, label="GNN", color="tab:blue")
    ax3.plot(times, mlp_energy_drift, linewidth=2, label="MLP", color="tab:orange")
    ax3.plot(
        times,
        true_energy_drift,
        linewidth=1.5,
        label="REBOUND",
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    ax3.set_xlabel("Time (years)", fontsize=11)
    ax3.set_ylabel("Relative Energy Drift (%)", fontsize=11)
    ax3.set_title("Energy Conservation", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    # Panel 4: Per-Body Position Error (GNN)
    ax4 = plt.subplot(2, 3, 4)
    for i, (name, color) in enumerate(zip(body_names, colors)):
        body_error = np.linalg.norm(
            gnn_positions[:, i, :] - true_positions[:, i, :], axis=1
        )
        ax4.plot(times, body_error, linewidth=2, label=name, color=color)

    ax4.set_xlabel("Time (years)", fontsize=11)
    ax4.set_ylabel("Position Error (AU)", fontsize=11)
    ax4.set_title("GNN Per-Body Position Error", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    # Panel 5: XY Trajectories (MLP)
    ax5 = plt.subplot(2, 3, 5)
    for i, (name, color) in enumerate(zip(body_names, colors)):
        ax5.plot(
            true_positions[:, i, 0],
            true_positions[:, i, 1],
            "-",
            color=color,
            linewidth=2,
            label=f"{name} (REBOUND)",
            alpha=0.8,
        )
        ax5.plot(
            mlp_positions[:, i, 0],
            mlp_positions[:, i, 1],
            "--",
            color=color,
            linewidth=1.5,
            label=f"{name} (MLP)",
            alpha=0.6,
        )

    ax5.scatter(
        [0],
        [0],
        s=300,
        c="gold",
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    ax5.set_xlabel("x (AU)", fontsize=11)
    ax5.set_ylabel("y (AU)", fontsize=11)
    ax5.set_title(
        "Orbital Trajectories (XY): MLP vs. Ground Truth",
        fontsize=12,
        fontweight="bold",
    )
    ax5.legend(fontsize=8, loc="upper right")
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect("equal")

    # Panel 6: Per-Body Position Error (MLP)
    ax6 = plt.subplot(2, 3, 6)
    for i, (name, color) in enumerate(zip(body_names, colors)):
        body_error = np.linalg.norm(
            mlp_positions[:, i, :] - true_positions[:, i, :], axis=1
        )
        ax6.plot(times, body_error, linewidth=2, label=name, color=color)

    ax6.set_xlabel("Time (years)", fontsize=11)
    ax6.set_ylabel("Position Error (AU)", fontsize=11)
    ax6.set_title("MLP Per-Body Position Error", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nRollout diagnostic plot saved to: {output_path}")


def main() -> None:
    """Main rollout comparison: GNN vs. MLP vs. REBOUND ground truth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running autoregressive rollout on device: {device}\n")

    print("Loading ground truth trajectory...")
    trajectory = load_trajectory(DATA_PATH)

    initial_positions = trajectory["positions"][0]
    initial_velocities = trajectory["velocities"][0]
    masses = trajectory["masses"]
    G = trajectory["G"]  # the REAL constant this trajectory was generated with

    print("Initial conditions:")
    print(f"  Masses: {masses}")
    print(f"  Initial positions shape: {initial_positions.shape}")
    print(f"  Initial velocities shape: {initial_velocities.shape}")

    print("\nLoading trained models...")
    normalizer = Normalizer.load(GNN_CHECKPOINT.parent / "normalizer.npz")
    sample_graph = trajectory_to_graphs(trajectory, normalizer=normalizer)[0]
    n_bodies = len(masses)

    mlp_model = MLPBaseline(
        input_dim=n_bodies * 7,
        hidden_dim=128,
        output_dim=n_bodies * 3,
        n_layers=2,
    ).to(device)
    mlp_model.load_state_dict(
        torch.load(MLP_CHECKPOINT, map_location=device, weights_only=True)
    )
    mlp_model.eval()

    gnn_model = GNNSurrogate(
        node_input_dim=sample_graph.x.shape[1],
        edge_input_dim=sample_graph.edge_attr.shape[1],
        hidden_dim=64,
        num_mp_layers=3,
    ).to(device)
    gnn_model.load_state_dict(
        torch.load(GNN_CHECKPOINT, map_location=device, weights_only=True)
    )
    gnn_model.eval()

    print("Models loaded successfully.")

    print(
        f"\nRunning {ROLLOUT_STEPS}-step rollout (simulating {ROLLOUT_STEPS * DT:.1f} years)..."
    )
    print("This may take a few minutes...\n")

    print("Rolling out GNN surrogate...")
    gnn_positions, gnn_velocities = rollout_trajectory(
        initial_positions,
        initial_velocities,
        masses,
        gnn_model,
        ROLLOUT_STEPS,
        DT,
        device,
        normalizer,
        G,
        is_gnn=True,
    )

    print("Rolling out MLP baseline...")
    mlp_positions, mlp_velocities = rollout_trajectory(
        initial_positions,
        initial_velocities,
        masses,
        mlp_model,
        ROLLOUT_STEPS,
        DT,
        device,
        normalizer,
        G,
        is_gnn=False,
    )

    true_positions = trajectory["positions"][: ROLLOUT_STEPS + 1]
    true_velocities = trajectory["velocities"][: ROLLOUT_STEPS + 1]

    print("\nComputing energy conservation metrics...")
    gnn_energies = np.array(
        [
            compute_energy(gnn_positions[i], gnn_velocities[i], masses, G)
            for i in range(ROLLOUT_STEPS + 1)
        ]
    )
    mlp_energies = np.array(
        [
            compute_energy(mlp_positions[i], mlp_velocities[i], masses, G)
            for i in range(ROLLOUT_STEPS + 1)
        ]
    )
    true_energies = np.array(
        [
            compute_energy(true_positions[i], true_velocities[i], masses, G)
            for i in range(ROLLOUT_STEPS + 1)
        ]
    )

    gnn_pos_errors = compute_trajectory_error(gnn_positions, true_positions)
    mlp_pos_errors = compute_trajectory_error(mlp_positions, true_positions)

    times = np.arange(ROLLOUT_STEPS + 1) * DT

    print("\n" + "=" * 70)
    print("               AUTOREGRESSIVE ROLLOUT RESULTS")
    print("=" * 70)
    print(
        f"Simulation Duration: {ROLLOUT_STEPS * DT:.1f} years ({ROLLOUT_STEPS} steps)"
    )
    print(f"Timestep: {DT} years\n")

    print(f"{'Metric':<35} | {'GNN':<15} | {'MLP':<15}")
    print("-" * 70)

    gnn_final_err = gnn_pos_errors[-1]
    mlp_final_err = mlp_pos_errors[-1]
    print(
        f"{'Final Mean Position Error (AU)':<35} | {gnn_final_err:<15.6f} | {mlp_final_err:<15.6f}"
    )

    gnn_mean_err = np.mean(gnn_pos_errors)
    mlp_mean_err = np.mean(mlp_pos_errors)
    print(
        f"{'Time-Averaged Position Error (AU)':<35} | {gnn_mean_err:<15.6f} | {mlp_mean_err:<15.6f}"
    )

    gnn_energy_drift = (
        np.abs(gnn_energies[-1] - gnn_energies[0]) / np.abs(gnn_energies[0]) * 100
    )
    mlp_energy_drift = (
        np.abs(mlp_energies[-1] - mlp_energies[0]) / np.abs(mlp_energies[0]) * 100
    )
    true_energy_drift = (
        np.abs(true_energies[-1] - true_energies[0]) / np.abs(true_energies[0]) * 100
    )
    print(
        f"{'Final Energy Drift (%)':<35} | {gnn_energy_drift:<15.4f} | {mlp_energy_drift:<15.4f}"
    )
    print(f"{'REBOUND Energy Drift (%)':<35} | {true_energy_drift:<15.4f} |")

    print("=" * 70)

    print("\nGenerating diagnostic plots...")
    plot_rollout_comparison(
        times,
        true_positions,
        gnn_positions,
        mlp_positions,
        gnn_energies,
        mlp_energies,
        true_energies,
        OUTPUT_DIR / "rollout_comparison.png",
    )

    print("\n✅ Phase 6 (Autoregressive Rollout) Complete!")
    print(f"📊 Check out the results: {OUTPUT_DIR / 'rollout_comparison.png'}")


if __name__ == "__main__":
    main()
