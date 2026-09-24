"""
train_gnn_residual.py
Trains the Residual GNN surrogate model.
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
from nbody_surrogate.residual_model import ResidualGNNSurrogate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Residual GNN on: {device}")

    # Load data
    print("Loading trajectory...")
    trajectory = load_trajectory("data/simulated/milestone_trajectory.npz")
    
    # Compute accelerations
    print("Computing accelerations...")
    accelerations = compute_gravitational_accelerations(
        trajectory["positions"], trajectory["masses"], G=trajectory["G"]
    )
    
    # Fit normalizer
    print("Fitting normalizer...")
    normalizer = Normalizer.fit(
        masses=trajectory["masses"],
        positions=trajectory["positions"],
        velocities=trajectory["velocities"],
        accelerations=accelerations,
    )
    
    # Convert to graphs
    print("Converting to graphs...")
    graphs = trajectory_to_graphs(trajectory, normalizer=normalizer)
    print(f"Created {len(graphs)} graph samples")

    n_samples = len(graphs)
    train_idx = int(0.70 * n_samples)
    val_idx = int(0.85 * n_samples)

    train_loader = DataLoader(graphs[:train_idx], batch_size=32, shuffle=True)
    val_loader = DataLoader(graphs[train_idx:val_idx], batch_size=32, shuffle=False)

    model = ResidualGNNSurrogate().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    checkpoint_dir = Path("checkpoints/residual_gnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(checkpoint_dir / "normalizer.npz")

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_a = model(batch)  # a_prior + residual_scale * GNN(state)
            loss = criterion(pred_a, batch.y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_train = total_loss / train_idx
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_a = model(batch)
                val_loss += criterion(pred_a, batch.y).item() * batch.num_graphs

        avg_val = val_loss / (val_idx - train_idx)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/100 | Train: {avg_train:.6f} | Val: {avg_val:.6f}"
            )

    print(f"✅ Residual GNN Training Complete! Best Val Loss: {best_val_loss:.6e}")

if __name__ == "__main__":
    train()
