"""
train_gnn_attention.py
Trains a Graph Attention Network (GAT) for N-body dynamics with interpretable attention weights.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from nbody_surrogate.attention_model import AttentionGNNSurrogate
from nbody_surrogate.dataset import (
    load_trajectory,
    trajectory_to_graphs,
    compute_gravitational_accelerations,
    Normalizer,
)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Attention GNN on: {device}")

    # Load data
    trajectory = load_trajectory("data/simulated/milestone_trajectory.npz")
    accelerations = compute_gravitational_accelerations(trajectory["positions"], trajectory["masses"])
    normalizer = Normalizer.fit(masses=trajectory["masses"], positions=trajectory["positions"], velocities=trajectory["velocities"], accelerations=accelerations)
    graphs = trajectory_to_graphs(trajectory)

    n_samples = len(graphs)
    train_idx = int(0.70 * n_samples)
    val_idx = int(0.85 * n_samples)

    print(f"Dataset: {n_samples} samples")
    print(f"Train: {train_idx}, Val: {val_idx - train_idx}, Test: {n_samples - val_idx}")

    train_loader = DataLoader(graphs[:train_idx], batch_size=32, shuffle=True)
    val_loader = DataLoader(graphs[train_idx:val_idx], batch_size=32, shuffle=False)

    # Create model
    model = AttentionGNNSurrogate(
        node_in_dim=7, edge_in_dim=4, hidden_dim=64, num_layers=3
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    checkpoint_dir = Path("checkpoints/attention_gnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print("\n" + "=" * 60)
    print("Starting Attention GNN Training...")
    print("=" * 60)

    for epoch in range(1, 101):
        # Training
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            loss = criterion(pred, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / train_idx
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += criterion(pred, batch.y).item() * batch.num_graphs

        avg_val_loss = val_loss / (val_idx - train_idx)
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  → Saved best model (val_loss: {avg_val_loss:.6f})")

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/100 | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

    print("\n" + "=" * 60)
    print("Attention GNN Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6e}")
    print("=" * 60)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Attention GNN Training Curves")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "training_curves.png", dpi=300)
    print(f"Saved training curves to {checkpoint_dir / 'training_curves.png'}")

if __name__ == "__main__":
    train()
