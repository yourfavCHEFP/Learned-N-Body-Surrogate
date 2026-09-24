"""Training script for MLP baseline model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from nbody_surrogate.baseline import MLPBaseline  # ← MISSING IMPORT!
from nbody_surrogate.dataset import (
    Normalizer,
    compute_gravitational_accelerations,
    load_trajectory,
    trajectory_to_graphs,  # ← Fixed: was trajectory_to_graph (missing 's')
)

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
HIDDEN_DIM = 128
N_LAYERS = 2

# Data parameters
DATA_PATH = Path("data/simulated/milestone_trajectory.npz")
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Output
CHECKPOINT_DIR = Path("checkpoints/baseline")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def flatten_graph_batch(batch):
    """
    Convert a batch of PyG Data objects into flat tensors for MLP.

    Args:
        batch: List of Data objects from trajectory_to_graphs

    Returns:
        inputs: (batch_size, N_bodies × 7) tensor
        targets: (batch_size, N_bodies × 3) tensor
    """
    inputs = []
    targets = []

    for graph in batch:
        flat_input = graph.x.flatten()
        flat_target = graph.y.flatten()
        inputs.append(flat_input)
        targets.append(flat_target)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return inputs, targets


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        inputs, targets = flatten_graph_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(model, data_loader, criterion, device):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = flatten_graph_batch(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def main():
    """Main training script."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trajectory
    print("Loading trajectory data...")
    trajectory = load_trajectory(DATA_PATH)

    # Compute accelerations
    print("Computing gravitational accelerations...")
    accelerations = compute_gravitational_accelerations(
        trajectory["positions"], trajectory["masses"], G=trajectory["G"]
    )

    # Create normalizer
    print("Fitting normalizer...")
    normalizer = Normalizer.fit(
        masses=trajectory["masses"],
        positions=trajectory["positions"],
        velocities=trajectory["velocities"],
        accelerations=accelerations,
    )
    normalizer.save(CHECKPOINT_DIR / "normalizer.npz")

    # Convert trajectory to graphs (now actually normalized)
    print("Converting trajectory to graphs...")
    graphs = trajectory_to_graphs(trajectory, normalizer=normalizer)
    print(f"Created {len(graphs)} graph samples")

    # Split dataset
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

    # Create DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x
    )

    # Determine dimensions
    sample_graph = graphs[0]
    n_bodies = sample_graph.x.shape[0]
    input_dim = n_bodies * 7
    output_dim = n_bodies * 3

    print("\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Output dim: {output_dim}")
    print(f"  Num layers: {N_LAYERS}")

    # Create model
    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
        n_layers=N_LAYERS,
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    train_losses = []
    val_losses = []
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

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training Progress", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "training_curves.png", dpi=150)
    print(f"\nSaved training curves to {CHECKPOINT_DIR / 'training_curves.png'}")

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
