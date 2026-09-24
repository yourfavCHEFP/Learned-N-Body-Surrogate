"""Learned N-Body Surrogate package.

Import layering matters here: sim.py and dataset.py are pure REBOUND/numpy
and should be usable (e.g. for Phase 1 physics work, or scripts that only
generate/inspect trajectories) without PyTorch installed at all. Everything
under this package used to import eagerly, which meant `import
nbody_surrogate` -- even just to reach `sim.py` -- hard-required torch. These
are now lazy (PEP 562 module __getattr__): a torch-backed class is only
imported the moment someone actually accesses it, e.g. `nbody_surrogate.GNNSurrogate`.
"""

from .dataset import (
    Normalizer,
    compute_gravitational_accelerations,
    load_trajectory,
    trajectory_to_graphs,
)
from .sim import milestone_system, run_simulation

__version__ = "0.1.0"

_LAZY_ATTRS = {
    "AttentionGNNSurrogate": ".attention_model",
    "MLPBaseline": ".baseline",
    "HamiltonianNN": ".hamiltonian_model",
    "GNNSurrogate": ".model",
    "InteractionLayer": ".model",
    "ResidualGNNSurrogate": ".residual_model",
    "symplectic_rollout": ".symplectic_integrator",
    "velocity_verlet_step": ".symplectic_integrator",
}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
