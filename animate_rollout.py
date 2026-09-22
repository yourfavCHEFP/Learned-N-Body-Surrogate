"""
Animated Rollout Visualization

Generates a side-by-side animated GIF comparing REBOUND ground truth
with GNN predictions over time.

Saves: checkpoints/rollout_animation.gif
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def load_rollout_data():
    """Load ground truth and GNN predictions."""

    # Load REBOUND ground truth
    data = np.load("data/simulated/milestone_trajectory.npz")
    rebound_pos = data["positions"]  # Shape: (2000, 3, 3)

    # Test set: timesteps 1400:1900 (500 steps)
    test_start = 1400
    test_end = 1900
    rebound_pos = rebound_pos[test_start:test_end]

    # Load GNN predictions
    try:
        gnn_data = np.load("checkpoints/gnn_rollout.npz")
        gnn_pos = gnn_data["positions"]
    except FileNotFoundError:
        print("GNN rollout data not found. Please run rollout.py first.")
        print("Generating predictions now...")

        # You'll need to modify rollout.py to save data
        # For now, use dummy data
        print("ERROR: Please modify rollout.py to save GNN predictions.")
        print("Add this at the end of rollout.py:")
        print("  np.savez('checkpoints/gnn_rollout.npz', positions=gnn_positions)")
        exit(1)

    return rebound_pos, gnn_pos


def create_animation(
    rebound_pos, gnn_pos, save_path="checkpoints/rollout_animation.gif"
):
    """
    Create side-by-side animation of REBOUND vs GNN.

    Args:
        rebound_pos: Ground truth positions (T, N_bodies, 3)
        gnn_pos: GNN predicted positions (T, N_bodies, 3)
        save_path: Where to save the GIF
    """

    n_steps = rebound_pos.shape[0]
    body_names = ["Star", "Inner Planet", "Outer Planet"]
    colors = ["gold", "dodgerblue", "orangered"]
    sizes = [300, 100, 150]  # Visual sizes for plotting

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Set up axes
    for ax, title in zip([ax1, ax2], ["REBOUND (Ground Truth)", "GNN Surrogate"]):
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel("X Position (AU)", fontsize=11)
        ax.set_ylabel("Y Position (AU)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    # Initialize scatter plots (bodies)
    reb_scatters = []
    gnn_scatters = []

    for i, (name, color, size) in enumerate(zip(body_names, colors, sizes)):
        # REBOUND bodies
        reb_scatter = ax1.scatter(
            [],
            [],
            s=size,
            color=color,
            edgecolors="black",
            linewidth=1.5,
            label=name,
            zorder=10,
        )
        reb_scatters.append(reb_scatter)

        # GNN bodies
        gnn_scatter = ax2.scatter(
            [],
            [],
            s=size,
            color=color,
            edgecolors="black",
            linewidth=1.5,
            label=name,
            zorder=10,
        )
        gnn_scatters.append(gnn_scatter)

    # Initialize trajectory lines (fading trails)
    trail_length = 50  # Number of previous positions to show
    reb_trails = [
        ax1.plot([], [], color=colors[i], alpha=0.3, linewidth=1)[0] for i in range(3)
    ]
    gnn_trails = [
        ax2.plot([], [], color=colors[i], alpha=0.3, linewidth=1)[0] for i in range(3)
    ]

    # Time text
    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=12, fontweight="bold")

    # Legends
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    def init():
        """Initialize animation."""
        for scatter in reb_scatters + gnn_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for trail in reb_trails + gnn_trails:
            trail.set_data([], [])
        time_text.set_text("")
        return reb_scatters + gnn_scatters + reb_trails + gnn_trails + [time_text]

    def animate(frame):
        """Update animation frame."""

        # Calculate time in years
        dt = 0.01  # Timestep in years
        time_years = frame * dt

        # Update bodies
        for i in range(3):
            # REBOUND
            reb_scatters[i].set_offsets(
                [[rebound_pos[frame, i, 0], rebound_pos[frame, i, 1]]]
            )

            # GNN
            gnn_scatters[i].set_offsets([[gnn_pos[frame, i, 0], gnn_pos[frame, i, 1]]])

            # Update trails
            start_idx = max(0, frame - trail_length)

            # REBOUND trail
            trail_x = rebound_pos[start_idx : frame + 1, i, 0]
            trail_y = rebound_pos[start_idx : frame + 1, i, 1]
            reb_trails[i].set_data(trail_x, trail_y)

            # GNN trail
            trail_x = gnn_pos[start_idx : frame + 1, i, 0]
            trail_y = gnn_pos[start_idx : frame + 1, i, 1]
            gnn_trails[i].set_data(trail_x, trail_y)

        # Update time text
        time_text.set_text(
            f"Time: {time_years:.2f} years  |  Step: {frame}/{n_steps-1}"
        )

        return reb_scatters + gnn_scatters + reb_trails + gnn_trails + [time_text]

    # Create animation
    print(f"Generating animation with {n_steps} frames...")
    print("This may take a few minutes...")

    # Sample every 5th frame to reduce file size (500 frames → 100 frames)
    frames = range(0, n_steps, 5)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=50, blit=True, repeat=True
    )

    # Save as GIF
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer="pillow", fps=20, dpi=100)

    print(f"✅ Animation saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading rollout data...")
    rebound_pos, gnn_pos = load_rollout_data()

    print(f"REBOUND positions shape: {rebound_pos.shape}")
    print(f"GNN positions shape: {gnn_pos.shape}")

    print("\nCreating animation...")
    print("Note: This will take 2-5 minutes depending on your system.")

    create_animation(rebound_pos, gnn_pos)

    print("\n✅ Done! Check checkpoints/rollout_animation.gif")
