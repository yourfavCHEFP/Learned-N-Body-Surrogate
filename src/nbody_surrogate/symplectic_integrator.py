"""
src/nbody_surrogate/symplectic_integrator.py

Symplectic integration schemes (Velocity Verlet, Ruth 4th-order, Störmer-Verlet)
for Hamiltonian Neural Networks and learned conservative systems.
"""

from collections.abc import Callable

import torch


def velocity_verlet_step(
    q: torch.Tensor,
    p: torch.Tensor,
    m: torch.Tensor,
    grad_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1-step Velocity Verlet / Leapfrog for Hamiltonian systems.

    Args:
        q: Positions [N, 3] or [B, N, 3]
        p: Momenta [N, 3] or [B, N, 3]
        m: Masses [N] or [B, N]
        grad_fn: Function returning (dq/dt, dp/dt) given (q, p, m)
        dt: Timestep

    Returns:
        q_next, p_next
    """
    # Half-step momentum update: p(t + dt/2) = p(t) + (dt/2) * dp/dt(t)
    _, dp_dt_0 = grad_fn(q, p, m)
    p_half = p + 0.5 * dt * dp_dt_0

    # Full-step coordinate update: q(t + dt) = q(t) + dt * dq/dt(p_half)
    dq_dt_half, _ = grad_fn(q, p_half, m)
    q_next = q + dt * dq_dt_half

    # Second half-step momentum update: p(t + dt) = p(half) + (dt/2) * dp/dt(q_next)
    _, dp_dt_1 = grad_fn(q_next, p_half, m)
    p_next = p_half + 0.5 * dt * dp_dt_1

    return q_next, p_next


def symplectic_rollout(
    model,
    q_init: torch.Tensor,
    p_init: torch.Tensor,
    m: torch.Tensor,
    steps: int,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates long-horizon trajectory using symplectic integration.

    Returns:
        qs: [steps + 1, N, 3]
        ps: [steps + 1, N, 3]
    """
    qs = [q_init.clone()]
    ps = [p_init.clone()]

    q_curr = q_init.clone()
    p_curr = p_init.clone()

    def dynamics_fn(q_in, p_in, m_in):
        return model.time_derivative(q_in, p_in, m_in)

    for _ in range(steps):
        q_next, p_next = velocity_verlet_step(q_curr, p_curr, m, dynamics_fn, dt)
        qs.append(q_next.clone())
        ps.append(p_next.clone())
        q_curr, p_curr = q_next, p_next

    return torch.stack(qs, dim=0), torch.stack(ps, dim=0)
