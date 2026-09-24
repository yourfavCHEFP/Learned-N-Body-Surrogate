"""One-step evaluation and head-to-head diagnostics: GNN vs. Baseline MLP."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from nbody_surrogate.baseline import MLPBaseline
from nbody_surrogate.dataset import (
    Normalizer,
    load_trajectory,
    trajectory_to_graphs,
)
from nbody_surrogate.model import GNNSurrogate

DATA_PATH = Path("data/simulated/milestone_trajectory.npz")
MLP_CHECKPOINT = Path("checkpoints/baseline/best_model.pt")
GNN_CHECKPOINT = Path("checkpoints/gnn/best_model.pt")
OUTPUT_DIR = Path("checkpoints")


def extract_mlp_predictions(
    baseline_model: torch.nn.Module,
    test_graphs: list,
    device: torch.device,
) -> np.ndarray:
    """Run one-step inference using the MLP Baseline."""
    baseline_model.eval()
    all_preds = []

    with torch.no_grad():
        for graph in test_graphs:
            flat_input = graph.x.flatten().unsqueeze(0).to(device)
            flat_pred = baseline_model(flat_input)
            pred_3d = flat_pred.view(-1, 3).cpu().numpy()
            all_preds.append(pred_3d)

    return np.vstack(all_preds)


def extract_gnn_predictions(
    gnn_model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one-step inference using the GNN Surrogate and extract targets."""
    gnn_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = gnn_model(batch).cpu().numpy()
            target = batch.y.cpu().numpy()

            all_preds.append(pred)
            all_targets.append(target)

    return np.vstack(all_preds), np.vstack(all_targets)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute MSE, MAE, Max Error, Relative Error (%), and R2."""
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    max_err = float(np.max(np.abs(preds - targets)))

    eps = 1e-8
    pred_norm = np.linalg.norm(preds, axis=1)
    target_norm = np.linalg.norm(targets, axis=1)
    rel_error = float(
        np.mean(np.abs(pred_norm - target_norm) / (target_norm + eps)) * 100.0
    )

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + eps)))

    return {
        "mse": mse,
        "mae": mae,
        "max_err": max_err,
        "rel_error_pct": rel_error,
        "r2": r2,
    }


def plot_diagnostics(
    targets: np.ndarray,
    mlp_preds: np.ndarray,
    gnn_preds: np.ndarray,
    output_path: Path,
) -> None:
    """Generate parity scatter plots and error distribution histograms."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    components = ["a_x", "a_y", "a_z"]
    for i, comp in enumerate(components):
        ax = axes[0, i]
        ax.scatter(
            targets[:, i],
            mlp_preds[:, i],
            alpha=0.4,
            s=12,
            label="MLP",
            color="tab:orange",
        )
        ax.scatter(
            targets[:, i],
            gnn_preds[:, i],
            alpha=0.4,
            s=12,
            label="GNN",
            color="tab:blue",
        )

        min_val = min(targets[:, i].min(), mlp_preds[:, i].min(), gnn_preds[:, i].min())
        max_val = max(targets[:, i].max(), mlp_preds[:, i].max(), gnn_preds[:, i].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            lw=1.5,
            label="Ground Truth (y=x)",
        )

        ax.set_title(f"Parity: {comp}")
        ax.set_xlabel("True Acceleration")
        ax.set_ylabel("Predicted Acceleration")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Error distribution histograms
    mlp_res = np.linalg.norm(mlp_preds - targets, axis=1)
    gnn_res = np.linalg.norm(gnn_preds - targets, axis=1)

    axes[1, 0].hist(mlp_res, bins=40, color="tab:orange", alpha=0.7, edgecolor="black")
    axes[1, 0].set_title(f"MLP Error Distribution (Mean: {mlp_res.mean():.6f})")
    axes[1, 0].set_xlabel("||a_pred - a_true||")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(gnn_res, bins=40, color="tab:blue", alpha=0.7, edgecolor="black")
    axes[1, 1].set_title(f"GNN Error Distribution (Mean: {gnn_res.mean():.6f})")
    axes[1, 1].set_xlabel("||a_pred - a_true||")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    # Per-body error comparison bar chart
    body_names = ["Star (Body 0)", "Planet 1 (Inner)", "Planet 2 (Outer)"]
    mlp_body_errs = [np.mean(mlp_res[i::3]) for i in range(3)]
    gnn_body_errs = [np.mean(gnn_res[i::3]) for i in range(3)]

    x = np.arange(len(body_names))
    width = 0.35
    axes[1, 2].bar(x - width / 2, mlp_body_errs, width, label="MLP", color="tab:orange")
    axes[1, 2].bar(x + width / 2, gnn_body_errs, width, label="GNN", color="tab:blue")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(body_names)
    axes[1, 2].set_title("Mean Error by Celestial Body")
    axes[1, 2].set_ylabel("Mean Euclidean Error")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nDiagnostic plot saved to: {output_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running One-Step Evaluation on device: {device}")

    # 1. Load dataset with exact same split seed
    trajectory = load_trajectory(DATA_PATH)
    normalizer = Normalizer.load(GNN_CHECKPOINT.parent / "normalizer.npz")
    graphs = trajectory_to_graphs(trajectory, normalizer=normalizer)

    n_samples = len(graphs)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    n_test = n_samples - n_train - n_val

    _, _, test_data = random_split(
        graphs,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    test_graphs = list(test_data)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # 2. Load MLP Baseline Checkpoint
    sample_graph = test_graphs[0]
    n_bodies = sample_graph.x.shape[0]
    mlp_model = MLPBaseline(
        input_dim=n_bodies * 7,
        hidden_dim=128,
        output_dim=n_bodies * 3,
        n_layers=2,
    ).to(device)
    mlp_model.load_state_dict(torch.load(MLP_CHECKPOINT, map_location=device))

    # 3. Load GNN Surrogate Checkpoint
    gnn_model = GNNSurrogate(
        node_input_dim=sample_graph.x.shape[1],
        edge_input_dim=sample_graph.edge_attr.shape[1],
        hidden_dim=64,
        num_mp_layers=3,
    ).to(device)
    gnn_model.load_state_dict(torch.load(GNN_CHECKPOINT, map_location=device))

    # 4. Inference
    gnn_preds, targets = extract_gnn_predictions(gnn_model, test_loader, device)
    mlp_preds = extract_mlp_predictions(mlp_model, test_graphs, device)

    # 5. Compute global metrics
    mlp_metrics = compute_metrics(mlp_preds, targets)
    gnn_metrics = compute_metrics(gnn_preds, targets)

    print("\n" + "=" * 65)
    print("           ONE-STEP HEAD-TO-HEAD TEST EVALUATION")
    print("=" * 65)
    print(f"{'Metric':<20} | {'MLP Baseline':<18} | {'GNN Surrogate':<18}")
    print("-" * 65)
    print(
        f"{'MSE Loss':<20} | {mlp_metrics['mse']:<18.8f} | {gnn_metrics['mse']:<18.8f}"
    )
    print(
        f"{'MAE Loss':<20} | {mlp_metrics['mae']:<18.8f} | {gnn_metrics['mae']:<18.8f}"
    )
    print(
        f"{'Max Error':<20} | {mlp_metrics['max_err']:<18.8f} | {gnn_metrics['max_err']:<18.8f}"
    )
    print(
        f"{'Mean Rel. Error (%)':<20} | {mlp_metrics['rel_error_pct']:<18.4f} | {gnn_metrics['rel_error_pct']:<18.4f}"
    )
    print(f"{'R² Score':<20} | {mlp_metrics['r2']:<18.6f} | {gnn_metrics['r2']:<18.6f}")
    print("=" * 65)

    # 6. Per-Body breakdown
    body_names = ["Star (Body 0)", "Planet 1 (Inner)", "Planet 2 (Outer)"]
    print("\nPer-Body MSE Breakdown:")
    for i, name in enumerate(body_names):
        mlp_b_mse = np.mean((mlp_preds[i::3] - targets[i::3]) ** 2)
        gnn_b_mse = np.mean((gnn_preds[i::3] - targets[i::3]) ** 2)
        print(f"  {name:<20}: MLP MSE = {mlp_b_mse:.8f} | GNN MSE = {gnn_b_mse:.8f}")

    # 7. Generate diagnostic plots
    plot_diagnostics(
        targets, mlp_preds, gnn_preds, OUTPUT_DIR / "evaluation_one_step.png"
    )


if __name__ == "__main__":
    main()
