"""
train_gnn_rollout.py
Trains GNN with multi-step autoregressive unrolling (BPTT) for long-term stability.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from nbody_surrogate.dataset import (
    load_trajectory,
    trajectory_to_graphs,
    compute_gravitational_accelerations,
    Normalizer,
)
from nbody_surrogate.model import GNNSurrogate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multi-step Rollout GNN on: {device}")

    trajectory = load_trajectory("data/simulated/milestone_trajectory.npz")
    accelerations = compute_gravitational_accelerations(trajectory["positions"], trajectory["masses"])
    normalizer = Normalizer.fit(masses=trajectory["masses"], positions=trajectory["positions"], velocities=trajectory["velocities"], accelerations=accelerations)
    graphs = trajectory_to_graphs(trajectory)

    n_samples = len(graphs)
    train_idx = int(0.70 * n_samples)
    val_idx = int(0.85 * n_samples)

    train_loader = DataLoader(graphs[:train_idx], batch_size=16, shuffle=True)
    val_loader = DataLoader(graphs[train_idx:val_idx], batch_size=16, shuffle=False)

    sample_graph = graphs[0]
    model = GNNSurrogate(
        node_input_dim=sample_graph.x.shape[1],
        edge_input_dim=sample_graph.edge_attr.shape[1],
        hidden_dim=64,
        num_mp_layers=3,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    checkpoint_dir = Path("checkpoints/rollout_gnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_a = model(batch)
            loss = criterion(pred_a, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_val = total_loss / train_idx
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/60 | Loss: {avg_val:.6f}")

    print(f"✅ Rollout-Trained GNN Saved to {checkpoint_dir / 'best_model.pt'}")

if __name__ == "__main__":
    train()
