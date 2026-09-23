"""
analyze_attention_patterns.py
Visualizes learned attention weights to see which gravitational interactions the model focuses on.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nbody_surrogate.attention_model import AttentionGNNSurrogate
from nbody_surrogate.dataset import (
    load_trajectory,
    trajectory_to_graphs,
    compute_gravitational_accelerations,
    Normalizer,
)

def analyze():
    device = torch.device("cpu")
    print("=" * 60)
    print("Analyzing Attention Patterns")
    print("=" * 60)

    # Load model
    checkpoint = Path("checkpoints/attention_gnn/best_model.pt")
    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint}")
        print("Run train_gnn_attention.py first!")
        return

    model = AttentionGNNSurrogate().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Load data
    trajectory = load_trajectory("data/simulated/milestone_trajectory.npz")
    accelerations = compute_gravitational_accelerations(trajectory["positions"], trajectory["masses"])
    normalizer = Normalizer.fit(masses=trajectory["masses"], positions=trajectory["positions"], velocities=trajectory["velocities"], accelerations=accelerations)
    graphs = trajectory_to_graphs(trajectory)
    
    masses = trajectory["masses"]
    names = ["Star", "Planet1", "Planet2"]  # Default names for 3-body system

    # Get one sample
    sample = graphs[1000].to(device)

    # Extract attention weights
    print("\nExtracting attention weights from first layer...")

    with torch.no_grad():
        # Forward pass and capture attention
        _ = model(sample)

        # Get attention weights from first layer
        if hasattr(model, "get_attention_weights"):
            attention = model.get_attention_weights(sample)
        else:
            print("⚠️  Model doesn't expose attention weights")
            print("Expected attention pattern:")
            print("  - HIGH attention: Star ↔ Planets")
            print("  - LOW attention: Planet ↔ Planet")
            return

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create attention matrix
    n_nodes = len(masses)
    att_matrix = np.zeros((n_nodes, n_nodes))

    edge_index = sample.edge_index.cpu().numpy()
    attention_np = attention.cpu().numpy()

    for i, (src, dst) in enumerate(edge_index.T):
        att_matrix[src, dst] = attention_np[i]

    im = ax.imshow(att_matrix, cmap="hot", interpolation="nearest")
    ax.set_xticks(range(n_nodes))
    ax.set_yticks(range(n_nodes))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_title("Learned Attention Weights\n(Brighter = Higher Attention)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "attention_patterns.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved attention visualization to {save_path}")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("  - Star-Planet edges should have HIGH attention")
    print("  - Planet-Planet edges should have LOW attention")
    print("  - This matches gravitational physics (star dominates)")
    print("=" * 60)

if __name__ == "__main__":
    analyze()
