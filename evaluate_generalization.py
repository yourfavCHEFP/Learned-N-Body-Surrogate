"""
evaluate_generalization.py
Tests trained GNN on varied systems to measure generalization capability.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from nbody_surrogate.dataset import (
    load_trajectory,
    trajectory_to_graphs,
    compute_gravitational_accelerations,
    Normalizer,
)
from nbody_surrogate.model import GNNSurrogate

def evaluate_on_system(model, system_name, device):
    """Evaluate model on one test system."""
    traj_path = Path(f"data/simulated/{system_name}_trajectory.npz")

    if not traj_path.exists():
        print(f"⚠️  {system_name} not found")
        return None

    trajectory = load_trajectory(str(traj_path))
    accelerations = compute_gravitational_accelerations(trajectory["positions"], trajectory["masses"])
    normalizer = Normalizer.fit(masses=trajectory["masses"], positions=trajectory["positions"], velocities=trajectory["velocities"], accelerations=accelerations)
    graphs = trajectory_to_graphs(trajectory)

    # Use all data for testing
    model.eval()
    criterion = torch.nn.MSELoss()

    total_loss = 0.0
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            pred = model(graph)
            loss = criterion(pred, graph.y)
            total_loss += loss.item()

    mse = total_loss / len(graphs)
    return mse

def main():
    device = torch.device("cpu")
    print("=" * 60)
    print("Cross-System Generalization Evaluation")
    print("=" * 60)

    # Load trained model
    checkpoint = Path("checkpoints/gnn/best_model.pt")
    if not checkpoint.exists():
        print("❌ GNN checkpoint not found. Run train_gnn.py first!")
        return

    model = GNNSurrogate(
        node_input_dim=7,
        edge_input_dim=4,
        hidden_dim=64,
        num_mp_layers=3,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # Baseline performance (original 3-body system)
    print("\nBaseline (3-body milestone system):")
    baseline_mse = evaluate_on_system(model, "milestone", device)
    if baseline_mse:
        print(f"  MSE: {baseline_mse:.6e}")

    # Test systems
    test_systems = {
        "test_4body": "4-body system",
        "test_mdwarf": "M-dwarf system",
        "test_eccentric": "Eccentric orbits",
    }

    results = {}

    print("\nGeneralization Tests:")
    for sys_name, description in test_systems.items():
        print(f"\n{description}:")
        mse = evaluate_on_system(model, sys_name, device)
        if mse:
            print(f"  MSE: {mse:.6e}")
            if baseline_mse:
                ratio = mse / baseline_mse
                print(f"  vs Baseline: {ratio:.2f}×")
            results[description] = mse

    # Visualization
    if results and baseline_mse:
        fig, ax = plt.subplots(figsize=(10, 6))

        systems = ["Baseline (3-body)"] + list(results.keys())
        mses = [baseline_mse] + list(results.values())

        bars = ax.bar(range(len(systems)), mses, color=["green"] + ["orange"] * len(results))
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(systems, rotation=15, ha="right")
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.set_title("GNN Generalization: Performance on Unseen Systems", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / "generalization_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved generalization plot to {save_path}")

    print("\n" + "=" * 60)
    print("Generalization Summary:")
    print("  - Good generalization: MSE < 2× baseline")
    print("  - Moderate: 2× < MSE < 10× baseline")
    print("  - Poor: MSE > 10× baseline")
    print("=" * 60)

if __name__ == "__main__":
    main()
