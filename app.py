"""Streamlit interface for exploring the learned N-body surrogate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rollout import rollout_trajectory  # noqa: E402

from nbody_surrogate.dataset import (  # noqa: E402
    Normalizer,
    Trajectory,
    compute_gravitational_accelerations,
    load_trajectory,
)
from nbody_surrogate.model import GNNSurrogate  # noqa: E402

DATA_DIR = ROOT / "data" / "simulated"
CHECKPOINT_DIR = ROOT / "checkpoints" / "gnn"
DEFAULT_DATASET = DATA_DIR / "milestone_trajectory.npz"
CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> Trajectory:
    return load_trajectory(path)


@st.cache_resource(show_spinner="Loading trained surrogate...")
def load_model(checkpoint_path: str, n_bodies: int) -> torch.nn.Module:
    device = torch.device("cpu")
    model = GNNSurrogate(
        node_input_dim=7,
        edge_input_dim=4,
        hidden_dim=64,
        num_mp_layers=3,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def fit_fallback_normalizer(path: str) -> Normalizer:
    trajectory = load_trajectory(path)
    accelerations = compute_gravitational_accelerations(
        trajectory["positions"], trajectory["masses"], G=trajectory["G"]
    )
    return Normalizer.fit(
        trajectory["masses"],
        trajectory["positions"],
        trajectory["velocities"],
        accelerations,
    )


def make_trajectory_plot(positions: np.ndarray, names: list[str]) -> go.Figure:
    figure = go.Figure()
    colors = ["#f3b63f", "#45a1d8", "#f06d5f", "#8c7ae6", "#55b88a"]
    for body_index, name in enumerate(names):
        figure.add_trace(
            go.Scatter3d(
                x=positions[:, body_index, 0],
                y=positions[:, body_index, 1],
                z=positions[:, body_index, 2],
                mode="lines",
                name=name,
                line={"color": colors[body_index % len(colors)], "width": 4},
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[positions[-1, body_index, 0]],
                y=[positions[-1, body_index, 1]],
                z=[positions[-1, body_index, 2]],
                mode="markers",
                name=f"{name} now",
                showlegend=False,
                marker={"color": colors[body_index % len(colors)], "size": 6},
            )
        )
    figure.update_layout(
        height=650,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "xaxis_title": "x (AU)",
            "yaxis_title": "y (AU)",
            "zaxis_title": "z (AU)",
            "aspectmode": "data",
        },
        legend={"orientation": "h", "y": 1.02},
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="N-Body Surrogate", page_icon="✦", layout="wide")
    st.title("Learned N-Body Surrogate")
    st.caption("Interactive CPU rollout of the trained graph neural network")

    if not CHECKPOINT.exists():
        st.error(f"Model checkpoint not found: {CHECKPOINT}")
        st.stop()

    datasets = sorted(DATA_DIR.glob("*.npz"))
    dataset_options = datasets or [DEFAULT_DATASET]
    default_index = (
        dataset_options.index(DEFAULT_DATASET)
        if DEFAULT_DATASET in dataset_options
        else 0
    )
    selected_path = st.sidebar.selectbox(
        "Initial system",
        dataset_options,
        index=default_index,
        format_func=lambda path: path.stem,
    )
    trajectory = load_dataset(str(selected_path))
    max_steps = min(500, len(trajectory["times"]) - 1)

    if len(trajectory["masses"]) != 3:
        st.sidebar.warning(
            "The checked-in GNN was trained on a 3-body system. This rollout is "
            "an out-of-distribution experiment, not a validated prediction."
        )

    steps = st.sidebar.slider(
        "Rollout steps", min_value=1, max_value=max_steps, value=min(100, max_steps)
    )
    dt = st.sidebar.number_input(
        "Timestep (years)",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f",
    )
    run_rollout = st.sidebar.button(
        "Run surrogate rollout", type="primary", use_container_width=True
    )

    if "rollout" not in st.session_state or run_rollout:
        normalizer_path = CHECKPOINT_DIR / "normalizer.npz"
        using_fallback = not normalizer_path.exists()
        normalizer = (
            fit_fallback_normalizer(str(selected_path))
            if using_fallback
            else Normalizer.load(normalizer_path)
        )
        model = load_model(str(CHECKPOINT), len(trajectory["masses"]))
        predicted_positions, predicted_velocities = rollout_trajectory(
            trajectory["positions"][0],
            trajectory["velocities"][0],
            trajectory["masses"],
            model,
            steps,
            float(dt),
            torch.device("cpu"),
            normalizer,
            trajectory["G"],
            is_gnn=True,
        )
        st.session_state.rollout = predicted_positions
        st.session_state.velocities = predicted_velocities
        st.session_state.dataset = selected_path.name
        st.session_state.fallback = using_fallback

    positions = st.session_state.rollout
    if st.session_state.dataset != selected_path.name:
        st.info("Choose Run surrogate rollout to apply the new initial system.")

    if st.session_state.fallback:
        st.warning(
            "This checkout has no checkpoint normalizer.npz. The app fitted a fallback "
            "normalizer from the selected trajectory; use the training normalizer for "
            "reproducible deployment results."
        )

    names = ["Star"] + [f"Body {index}" for index in range(1, positions.shape[1])]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Bodies", positions.shape[1])
    metric_columns[1].metric("Steps", positions.shape[0] - 1)
    metric_columns[2].metric(
        "Simulated time", f"{(positions.shape[0] - 1) * dt:.2f} yr"
    )
    st.plotly_chart(make_trajectory_plot(positions, names), width="stretch")


if __name__ == "__main__":
    main()
