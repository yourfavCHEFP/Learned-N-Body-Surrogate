"""
train_gnn_rollout.py
Trains GNN with GENUINE multi-step autoregressive unrolling (BPTT) for
long-term rollout stability.

Previously this file was misnamed: despite the docstring, the training loop
was identical single-step supervision to train_gnn.py (predict acceleration
at t, compare to ground truth at t, no rollout, no BPTT), and its val_loader
was built but never used anywhere -- "best model" was actually selected by
TRAINING loss.

This version:
  1. Samples random K-step windows from the trajectory.
  2. Rolls the model out autoregressively for K steps (symplectic Euler --
     see note below), entirely in torch so gradients flow back through the
     whole window (that's what "BPTT" means here).
  3. Backprops a loss over the WHOLE rolled-out window against the real
     REBOUND positions/velocities, not just a one-step acceleration target.
     This is what actually teaches the model to resist compounding rollout
     error, which one-step training does not do.
  4. Uses a genuine held-out multi-step validation rollout (no_grad) for
     checkpoint selection.

Integrator note: this uses symplectic (semi-implicit) Euler at TRAIN time
(one model call per step: v += a*dt; x += v*dt), not the velocity-Verlet
used for final evaluation in rollout.py (which needs two model calls per
step). This is a deliberate simplicity/speed tradeoff during BPTT training,
common in this literature -- both are consistent first/second-order
integrators of the same learned dynamics, so training under one and
evaluating under the other is fine, just worth knowing about.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data

from nbody_surrogate.dataset import (
    Normalizer,
    compute_gravitational_accelerations,
    load_trajectory,
    trajectory_to_graphs,
    _complete_graph_edges,
)
from nbody_surrogate.model import GNNSurrogate

WINDOW_LEN = 8          # K: number of autoregressive steps per BPTT window
BATCH_WINDOWS = 16      # B: number of independent windows per gradient step
ITERS_PER_EPOCH = 40
EPOCHS = 60
VAL_WINDOWS = 24        # held-out windows used for checkpoint selection
VAL_WINDOW_LEN = 16     # deliberately longer than train windows: also checks
                         # a bit of generalization to a longer horizon


def _batched_edge_index(n_bodies: int, batch_size: int, device) -> torch.Tensor:
    """Tile the complete-graph edge_index across `batch_size` independent
    graphs, offsetting node indices per graph so they don't collide."""
    base = _complete_graph_edges(n_bodies).to(device)  # (2, E)
    E = base.shape[1]
    offsets = (torch.arange(batch_size, device=device) * n_bodies).view(-1, 1, 1)
    tiled = base.unsqueeze(0).expand(batch_size, -1, -1) + offsets  # (B,2,E)
    return tiled.permute(1, 0, 2).reshape(2, batch_size * E), E


def _torch_normalize(value, mean, std):
    return (value - mean) / std


def _torch_denormalize_acc(value, mean, std):
    return value * std + mean


class TorchRolloutHelper:
    """Holds normalizer stats as torch tensors and builds batched graphs for
    a differentiable multi-step rollout."""

    def __init__(self, normalizer: Normalizer, masses: np.ndarray, G: float, device):
        self.device = device
        self.G = G
        self.n_bodies = len(masses)
        self.masses_np = masses

        f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        self.mass_mean, self.mass_std = float(normalizer.mass_mean), float(normalizer.mass_std)
        self.pos_mean, self.pos_std = f(normalizer.pos_mean), f(normalizer.pos_std)
        self.vel_mean, self.vel_std = f(normalizer.vel_mean), f(normalizer.vel_std)
        self.acc_mean, self.acc_std = f(normalizer.acc_mean), f(normalizer.acc_std)

        norm_mass_np = normalizer.normalize_mass(masses)  # (n_bodies,)
        self.masses_t = f(masses)  # physical masses, (n_bodies,)
        self.norm_mass_1 = f(norm_mass_np)  # (n_bodies,)

    def normalize_mass_batch(self, batch_size: int) -> torch.Tensor:
        return self.norm_mass_1.repeat(batch_size).unsqueeze(-1)  # (B*n_bodies, 1)

    def normalize_pos(self, pos):
        return _torch_normalize(pos, self.pos_mean, self.pos_std)

    def normalize_vel(self, vel):
        return _torch_normalize(vel, self.vel_mean, self.vel_std)

    def denormalize_acc(self, acc_norm):
        return _torch_denormalize_acc(acc_norm, self.acc_mean, self.acc_std)

    def build_batch(self, pos, vel, edge_index_batched, edge_src_local, edge_tgt_local, batch_size):
        """
        pos, vel: (B, n_bodies, 3) physical-units tensors.
        Returns a Data-like object the GNN can consume directly.
        """
        n = self.n_bodies
        x = torch.cat(
            [
                self.normalize_mass_batch(batch_size),
                self.normalize_pos(pos).reshape(batch_size * n, 3),
                self.normalize_vel(vel).reshape(batch_size * n, 3),
            ],
            dim=-1,
        )

        # Edge geometry per graph in the batch (relative pos + distance),
        # gathered via the LOCAL (0..n_bodies-1) source/target indices,
        # then normalized the same way trajectory_to_graphs does.
        src = pos[:, edge_src_local, :]  # (B, E, 3)
        tgt = pos[:, edge_tgt_local, :]  # (B, E, 3)
        rel = tgt - src
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        rel_norm = rel / self.pos_std
        dist_norm = dist / self.pos_std.mean()
        edge_attr = torch.cat([rel_norm, dist_norm], dim=-1).reshape(-1, 4)

        return Data(x=x, edge_index=edge_index_batched, edge_attr=edge_attr)

    def rollout_step(self, model, pos, vel, edge_index_batched, edge_src_local, edge_tgt_local, batch_size, dt):
        """One symplectic-Euler step, fully differentiable."""
        graph = self.build_batch(pos, vel, edge_index_batched, edge_src_local, edge_tgt_local, batch_size)
        acc_norm = model(graph).reshape(batch_size, self.n_bodies, 3)
        acc = self.denormalize_acc(acc_norm)
        new_vel = vel + acc * dt
        new_pos = pos + new_vel * dt
        return new_pos, new_vel


def _sample_windows(positions, velocities, start_lo, start_hi, window_len, batch_size, rng):
    """Sample `batch_size` random start indices in [start_lo, start_hi) and
    return (init_pos, init_vel, target_pos, target_vel) tensors."""
    starts = rng.integers(start_lo, start_hi - window_len, size=batch_size)
    init_pos = np.stack([positions[s] for s in starts])
    init_vel = np.stack([velocities[s] for s in starts])
    target_pos = np.stack([positions[s + 1 : s + 1 + window_len] for s in starts])  # (B, K, n, 3)
    target_vel = np.stack([velocities[s + 1 : s + 1 + window_len] for s in starts])
    return init_pos, init_vel, target_pos, target_vel


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multi-step Rollout GNN (real BPTT) on: {device}")

    trajectory = load_trajectory("data/simulated/milestone_trajectory.npz")
    G = trajectory["G"]
    masses = trajectory["masses"]
    positions = trajectory["positions"]
    velocities = trajectory["velocities"]
    times = trajectory["times"]
    dt = float(times[1] - times[0])
    n_bodies = len(masses)

    accelerations = compute_gravitational_accelerations(positions, masses, G=G)
    normalizer = Normalizer.fit(masses=masses, positions=positions, velocities=velocities, accelerations=accelerations)

    n_snapshots = len(times)
    train_idx = int(0.70 * n_snapshots)
    val_idx = int(0.85 * n_snapshots)

    checkpoint_dir = Path("checkpoints/rollout_gnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(checkpoint_dir / "normalizer.npz")

    # Use one throwaway one-step graph purely to read off feature dims.
    sample_graph = trajectory_to_graphs(trajectory, normalizer=normalizer)[0]
    model = GNNSurrogate(
        node_input_dim=sample_graph.x.shape[1],
        edge_input_dim=sample_graph.edge_attr.shape[1],
        hidden_dim=64,
        num_mp_layers=3,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    helper = TorchRolloutHelper(normalizer, masses, G, device)

    base_edges = _complete_graph_edges(n_bodies)
    edge_src_local = base_edges[0].numpy()
    edge_tgt_local = base_edges[1].numpy()

    train_edge_index, _ = _batched_edge_index(n_bodies, BATCH_WINDOWS, device)
    val_edge_index, _ = _batched_edge_index(n_bodies, VAL_WINDOWS, device)
    edge_src_local_t = torch.as_tensor(edge_src_local, dtype=torch.long, device=device)
    edge_tgt_local_t = torch.as_tensor(edge_tgt_local, dtype=torch.long, device=device)

    rng = np.random.default_rng(0)

    # Fixed validation windows (sampled once, reused every epoch) so
    # checkpoint selection is comparing apples to apples across epochs.
    val_rng = np.random.default_rng(42)
    val_init_pos, val_init_vel, val_target_pos, val_target_vel = _sample_windows(
        positions, velocities, train_idx, val_idx, VAL_WINDOW_LEN, VAL_WINDOWS, val_rng
    )
    val_init_pos_t = torch.tensor(val_init_pos, dtype=torch.float32, device=device)
    val_init_vel_t = torch.tensor(val_init_vel, dtype=torch.float32, device=device)
    val_target_pos_t = torch.tensor(val_target_pos, dtype=torch.float32, device=device)
    val_target_vel_t = torch.tensor(val_target_vel, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    print(
        f"Windows: train K={WINDOW_LEN} x {BATCH_WINDOWS}/iter x {ITERS_PER_EPOCH} iters/epoch | "
        f"val K={VAL_WINDOW_LEN} x {VAL_WINDOWS} (fixed)"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(ITERS_PER_EPOCH):
            init_pos, init_vel, target_pos, target_vel = _sample_windows(
                positions, velocities, 0, train_idx, WINDOW_LEN, BATCH_WINDOWS, rng
            )
            pos = torch.tensor(init_pos, dtype=torch.float32, device=device)
            vel = torch.tensor(init_vel, dtype=torch.float32, device=device)
            tgt_pos = torch.tensor(target_pos, dtype=torch.float32, device=device)
            tgt_vel = torch.tensor(target_vel, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            loss = 0.0
            for step in range(WINDOW_LEN):
                pos, vel = helper.rollout_step(
                    model, pos, vel, train_edge_index, edge_src_local_t, edge_tgt_local_t, BATCH_WINDOWS, dt
                )
                loss = loss + criterion(pos, tgt_pos[:, step]) + criterion(vel, tgt_vel[:, step])
            loss = loss / WINDOW_LEN

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / ITERS_PER_EPOCH

        # --- Real multi-step validation rollout, no_grad, longer horizon ---
        model.eval()
        with torch.no_grad():
            pos, vel = val_init_pos_t, val_init_vel_t
            val_loss = 0.0
            for step in range(VAL_WINDOW_LEN):
                pos, vel = helper.rollout_step(
                    model, pos, vel, val_edge_index, edge_src_local_t, edge_tgt_local_t, VAL_WINDOWS, dt
                )
                val_loss += criterion(pos, val_target_pos_t[:, step]).item()
                val_loss += criterion(vel, val_target_vel_t[:, step]).item()
            val_loss /= VAL_WINDOW_LEN

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | Train BPTT loss: {avg_train_loss:.6f} | "
                f"Val ({VAL_WINDOW_LEN}-step) loss: {val_loss:.6f}"
                f"{'  <- best' if val_loss == best_val_loss else ''}"
            )

    print(f"✅ Rollout-Trained GNN (real multi-step BPTT) saved to {checkpoint_dir / 'best_model.pt'}")
    print(f"   Best {VAL_WINDOW_LEN}-step validation loss: {best_val_loss:.6e}")


if __name__ == "__main__":
    train()
