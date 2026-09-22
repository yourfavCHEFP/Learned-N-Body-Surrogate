"""
3D Trajectory Visualization

Generates a static 3D plot comparing REBOUND ground truth, GNN predictions,
and MLP predictions over the entire rollout trajectory.

Saves: checkpoints/trajectory_3d_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np


def load_rollout_data():
    """Load ground truth and model predictions from rollout results."""

    # Load REBOUND ground truth
    data = np.load("data/simulated/milestone_trajectory.npz")
    rebound_pos = data["positions"]  # Shape: (2000, 3, 3)

    # For rollout, we use timesteps 1400:1900 (test set, 500 steps)
    test_start = 1400
    test_end = 1900
    rebound_pos = rebound_pos[test_start:test_end]

    # Load model predictions (saved during rollout.py execution)
    # If rollout.py doesn't save these, we'll regenerate them
    try:
        gnn_data = np.load("checkpoints/gnn_rollout.npz")
        mlp_data = np.load("checkpoints/mlp_rollout.npz")

        gnn_pos = gnn_data["positions"]
        mlp_pos = mlp_data["positions"]

    except FileNotFoundError:
        print("Rollout data not found. Please run rollout.py first.")
        print("Generating predictions now...")

        # Import and run rollout
        from rollout import perform_rollout

        gnn_pos, mlp_pos = perform_rollout(save_data=True)

    return rebound_pos, gnn_pos, mlp_pos


def plot_3d_trajectories(
    rebound_pos, gnn_pos, mlp_pos, save_path="checkpoints/trajectory_3d_comparison.png"
):
    """
    Create 3D visualization of orbital trajectories.

    Args:
        rebound_pos: Ground truth positions (T, N_bodies, 3)
        gnn_pos: GNN predicted positions (T, N_bodies, 3)
        mlp_pos: MLP predicted positions (T, N_bodies, 3)
        save_path: Where to save the figure
    """

    n_bodies = rebound_pos.shape[1]
    body_names = ["Star", "Inner Planet", "Outer Planet"]
    colors = ["gold", "dodgerblue", "orangered"]

    # Create figure with 3 subplots (one per body)
    fig = plt.figure(figsize=(20, 6))

    for body_idx in range(n_bodies):
        ax = fig.add_subplot(1, 3, body_idx + 1, projection="3d")

        # Extract positions for this body
        reb_x = rebound_pos[:, body_idx, 0]
        reb_y = rebound_pos[:, body_idx, 1]
        reb_z = rebound_pos[:, body_idx, 2]

        gnn_x = gnn_pos[:, body_idx, 0]
        gnn_y = gnn_pos[:, body_idx, 1]
        gnn_z = gnn_pos[:, body_idx, 2]

        mlp_x = mlp_pos[:, body_idx, 0]
        mlp_y = mlp_pos[:, body_idx, 1]
        mlp_z = mlp_pos[:, body_idx, 2]

        # Plot trajectories
        ax.plot(
            reb_x,
            reb_y,
            reb_z,
            color="black",
            linewidth=2,
            alpha=0.8,
            label="REBOUND (Ground Truth)",
        )

        ax.plot(
            gnn_x,
            gnn_y,
            gnn_z,
            color="green",
            linewidth=1.5,
            alpha=0.7,
            label="GNN Surrogate",
        )

        ax.plot(
            mlp_x,
            mlp_y,
            mlp_z,
            color="red",
            linewidth=1.5,
            alpha=0.7,
            linestyle="--",
            label="MLP Baseline",
        )

        # Mark start and end points
        ax.scatter(
            reb_x[0],
            reb_y[0],
            reb_z[0],
            color=colors[body_idx],
            s=100,
            marker="o",
            edgecolors="black",
            linewidth=2,
            label="Start",
        )

        ax.scatter(
            reb_x[-1],
            reb_y[-1],
            reb_z[-1],
            color=colors[body_idx],
            s=100,
            marker="s",
            edgecolors="black",
            linewidth=2,
            label="End",
        )

        # Labels and title
        ax.set_xlabel("X Position (AU)", fontsize=11)
        ax.set_ylabel("Y Position (AU)", fontsize=11)
        ax.set_zlabel("Z Position (AU)", fontsize=11)
        ax.set_title(
            f"{body_names[body_idx]} Trajectory", fontsize=13, fontweight="bold"
        )

        # Equal aspect ratio
        max_range = np.max([np.ptp(reb_x), np.ptp(reb_y), np.ptp(reb_z)])

        mid_x = (reb_x.max() + reb_x.min()) / 2
        mid_y = (reb_y.max() + reb_y.min()) / 2
        mid_z = (reb_z.max() + reb_z.min()) / 2

        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # Legend (only on first subplot)
        if body_idx == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # Adjust viewing angle
        ax.view_init(elev=20, azim=45)

        # Grid
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "3D Orbital Trajectories: 500-Step Rollout Comparison",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved 3D trajectory visualization to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading rollout data...")
    rebound_pos, gnn_pos, mlp_pos = load_rollout_data()

    print(f"REBOUND positions shape: {rebound_pos.shape}")
    print(f"GNN positions shape: {gnn_pos.shape}")
    print(f"MLP positions shape: {mlp_pos.shape}")

    print("\nGenerating 3D visualization...")
    plot_3d_trajectories(rebound_pos, gnn_pos, mlp_pos)

    print("\n✅ Done! Check checkpoints/trajectory_3d_comparison.png")
