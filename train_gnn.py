"""Training script for Graph Neural Network (GNN) surrogate model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from nbody_surrogate.dataset import (
    Normalizer,
    compute_gravitational_accelerations,
    load_trajectory,
    trajectory_to_graphs,
)
from nbody_surrogate.model import GNNSurrogate

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
HIDDEN_DIM = 64
NUM_MP_LAYERS = 3

# Data parameters
DATA_PATH = Path("data/simulated/milestone_trajectory.npz")
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Output directory for GNN checkpoints
CHECKPOINT_DIR = Path("checkpoints/gnn")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the GNN model for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the GNN model on validation/test data."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def main() -> None:
    """Main training script for GNN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load trajectory
    print("Loading trajectory data...")
    trajectory = load_trajectory(DATA_PATH)

    # 2. Compute accelerations
    print("Computing gravitational accelerations...")
    accelerations = compute_gravitational_accelerations(
        trajectory["positions"], trajectory["masses"], G=trajectory["G"]
    )

    # 3. Fit normalizer
    print("Fitting normalizer...")
    normalizer = Normalizer.fit(
        masses=trajectory["masses"],
        positions=trajectory["positions"],
        velocities=trajectory["velocities"],
        accelerations=accelerations,
    )
    normalizer.save(CHECKPOINT_DIR / "normalizer.npz")

    # 4. Convert trajectory to graphs (normalized -- this is what actually
    # makes the normalizer's stats reach the model; it used to be computed
    # and then silently discarded here).
    print("Converting trajectory to graphs...")
    graphs = trajectory_to_graphs(trajectory, normalizer=normalizer)
    print(f"Created {len(graphs)} graph samples")

    # 5. Split dataset
    n_samples = len(graphs)
    n_train = int(n_samples * TRAIN_SPLIT)
    n_val = int(n_samples * VAL_SPLIT)
    n_test = n_samples - n_train - n_val

    train_data, val_data, test_data = random_split(
        graphs,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # 6. Create PyG DataLoaders (automatic graph batching)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 7. Create GNN Model
    sample_graph = graphs[0]
    node_dim = sample_graph.x.shape[1]  # 7
    edge_dim = sample_graph.edge_attr.shape[1]  # 4

    print("\nGNN architecture:")
    print(f"  Node Input dim: {node_dim}")
    print(f"  Edge Input dim: {edge_dim}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Num MP layers: {NUM_MP_LAYERS}")

    model = GNNSurrogate(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_mp_layers=NUM_MP_LAYERS,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 8. Training loop
    print("\n" + "=" * 50)
    print("Starting GNN training...")
    print("=" * 50)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 50)

    # 9. Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("GNN Training Progress", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "training_curves.png", dpi=150)
    print(f"\nSaved training curves to {CHECKPOINT_DIR / 'training_curves.png'}")

    # 10. Final test evaluation
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
