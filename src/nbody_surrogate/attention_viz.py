"""
src/nbody_surrogate/attention_viz.py

Visualization utility functions for graph attention weights.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_matrix(
    attention_weights: torch.Tensor,
    edge_index: torch.Tensor,
    body_names: list[str],
    save_path: str | Path = "results/attention_matrix.png",
    title: str = "Learned Gravitational Attention",
):
    """
    Plots an N x N heatmap of directed attention weights between celestial bodies.
    """
    n_bodies = len(body_names)
    matrix = np.zeros((n_bodies, n_bodies))

    edge_index_np = edge_index.detach().cpu().numpy()
    weights_np = attention_weights.detach().cpu().numpy().flatten()

    for i in range(edge_index_np.shape[1]):
        src = edge_index_np[0, i]
        dst = edge_index_np[1, i]
        matrix[dst, src] = weights_np[i]

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(matrix, cmap="viridis", alpha=0.9)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")

    ax.set_xticks(range(n_bodies))
    ax.set_yticks(range(n_bodies))
    ax.set_xticklabels(body_names, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(body_names, fontsize=10)
    ax.set_xlabel("Source Body (j)", fontsize=11, labelpad=10)
    ax.set_ylabel("Target Body (i)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    # Annotate values
    for i in range(n_bodies):
        for j in range(n_bodies):
            val = matrix[i, j]
            color = "white" if val < (matrix.max() / 2) else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9
            )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved attention matrix plot to: {save_path}")
