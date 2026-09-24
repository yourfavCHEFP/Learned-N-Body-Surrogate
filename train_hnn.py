"""
train_hnn.py
Trains the Hamiltonian Neural Network for energy-conserving dynamics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from nbody_surrogate.dataset import load_trajectory
from nbody_surrogate.hamiltonian_model import HamiltonianNN

def prepare_hnn_data(positions, velocities, masses):
    """
    Convert trajectory data to (q, p, m) format for HNN.

    Args:
        positions: (T, N, 3) positions
        velocities: (T, N, 3) velocities
        masses: (N,) masses

    Returns:
        q_data: (T, N, 3) positions
        p_data: (T, N, 3) momenta (p = m*v)
        m_data: (N,) masses
    """
    T, N, _ = positions.shape

    # Convert velocities to momenta: p = m*v
    momenta = velocities * masses[None, :, None]  # (T, N, 3)

    return positions, momenta, masses

def compute_ground_truth_derivatives(positions, velocities, dt):
    """
    Compute ground truth time derivatives from finite differences.

    Args:
        positions: (T, N, 3)
        velocities: (T, N, 3)
        dt: timestep

    Returns:
        dq_dt_true: (T-1, N, 3) velocity at each step
        dp_dt_true: (T-1, N, 3) force at each step
    """
    # dq/dt = velocity (already have this)
    dq_dt = velocities[:-1]  # (T-1, N, 3)

    # dp/dt = force = m*a (compute from momentum differences)
    dp_dt = np.diff(velocities, axis=0) / dt  # (T-1, N, 3)

    return dq_dt, dp_dt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Hamiltonian Neural Network on: {device}")

    # Load trajectory data
    trajectory = load_trajectory("data/simulated/milestone_trajectory.npz")
    
    times = trajectory["times"]
    pos = trajectory["positions"]
    vel = trajectory["velocities"]
    masses = trajectory["masses"]
    
    dt = times[1] - times[0]
    n_bodies = len(masses)

    print(f"Loaded trajectory: {len(times)} timesteps, {n_bodies} bodies")

    # Prepare HNN data (positions, momenta, masses)
    q_data, p_data, m_data = prepare_hnn_data(pos, vel, masses)

    # Compute ground truth derivatives
    dq_dt_true, dp_dt_true = compute_ground_truth_derivatives(
        pos, vel * masses[:, None], dt
    )

    # Convert to torch tensors
    q_train = torch.tensor(q_data[:-1], dtype=torch.float32)  # (T-1, N, 3)
    p_train = torch.tensor(p_data[:-1], dtype=torch.float32)  # (T-1, N, 3)
    m_train = torch.tensor(m_data, dtype=torch.float32)  # (N,)

    dq_target = torch.tensor(dq_dt_true, dtype=torch.float32)  # (T-1, N, 3)
    dp_target = torch.tensor(dp_dt_true, dtype=torch.float32)  # (T-1, N, 3)

    # Split train/val
    n_samples = len(q_train)
    train_idx = int(0.7 * n_samples)
    val_idx = int(0.85 * n_samples)

    print(f"Train: {train_idx}, Val: {val_idx - train_idx}, Test: {n_samples - val_idx}")

    # Create model
    model = HamiltonianNN(n_bodies=n_bodies, hidden_dim=128).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    checkpoint_dir = Path("checkpoints/hnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print("\n" + "=" * 60)
    print("Starting HNN Training...")
    print("=" * 60)

    for epoch in range(1, 101):
        model.train()

        # Training loop
        total_loss = 0.0
        batch_size = 32
        n_batches = train_idx // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            q_batch = q_train[start:end].to(device)
            p_batch = p_train[start:end].to(device)
            m_batch = m_train.unsqueeze(0).expand(batch_size, -1).to(device)

            dq_batch = dq_target[start:end].to(device)
            dp_batch = dp_target[start:end].to(device)

            optimizer.zero_grad()

            # Predict derivatives using Hamiltonian
            dq_pred, dp_pred = model.time_derivative(q_batch, p_batch, m_batch)

            # Loss: Match predicted dynamics to ground truth
            loss_q = torch.mean((dq_pred - dq_batch) ** 2)
            loss_p = torch.mean((dp_pred - dp_batch) ** 2)
            loss = loss_q + loss_p

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation. NOTE: this deliberately does NOT use torch.no_grad().
        # time_derivative() computes dH/dq, dH/dp via torch.autograd.grad,
        # which needs gradient tracking enabled for q/p even though we never
        # call .backward() or optimizer.step() here -- no_grad() previously
        # made that impossible, which is why this used to be faked as
        # `avg_train_loss * 1.1`. model.eval() still correctly disables any
        # dropout/batchnorm-style behavior; it just doesn't disable autograd.
        model.eval()
        q_val = q_train[train_idx:val_idx].to(device)
        p_val = p_train[train_idx:val_idx].to(device)
        m_val = m_train.unsqueeze(0).expand(val_idx - train_idx, -1).to(device)

        dq_val = dq_target[train_idx:val_idx].to(device)
        dp_val = dp_target[train_idx:val_idx].to(device)

        dq_pred, dp_pred = model.time_derivative(q_val, p_val, m_val)
        val_loss_q = torch.mean((dq_pred - dq_val) ** 2)
        val_loss_p = torch.mean((dp_pred - dp_val) ** 2)
        val_loss = (val_loss_q + val_loss_p).item()

        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            if epoch % 10 == 0:
                print(f"  → Saved best model (val_loss: {val_loss:.6f})")

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/100 | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print("\n" + "=" * 60)
    print("HNN Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6e}")
    print("=" * 60)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", alpha=0.7)
    plt.plot(val_losses, label="Val Loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Hamiltonian Neural Network Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    print(f"Saved training curves to {checkpoint_dir / 'training_curves.png'}")

if __name__ == "__main__":
    train()
