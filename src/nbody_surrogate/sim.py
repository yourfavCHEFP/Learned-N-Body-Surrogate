"""
sim.py — Phase 1: the REBOUND ground-truth simulator.

This module is the "teacher" for the whole project. Everything downstream
(dataset generation, the GNN, evaluation) trusts whatever this file produces,
so its only job right now is: generate small, physically sensible N-body
systems and integrate them forward in time using REBOUND's WHFast symplectic
integrator.

Units: astronomical units throughout (§2.5 of the project guide).
    - distance : AU
    - mass     : solar masses (M_sun)
    - time     : years
    - G        : NOT 1.0 in these units -- REBOUND computes G = 4*pi**2 (~39.48)
                 for ("yr","AU","Msun"). Every SimResult carries its own `G`
                 (read from `sim.G`); always use that value downstream instead
                 of assuming a constant.

Nothing in this file touches PyTorch / PyTorch Geometric. That's deliberate —
Phase 1 is "can we trust the physics," not "can we train a model."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rebound


@dataclass
class BodySpec:
    """Initial conditions for one body, in astro units."""

    name: str
    mass: float  # solar masses
    a: float | None = None  # semi-major axis (AU); None => central body
    e: float = 0.0  # eccentricity
    inc: float = 0.0  # inclination (radians)


@dataclass
class SimResult:
    """Container for a single simulated trajectory."""

    times: np.ndarray  # shape (T,)
    positions: np.ndarray  # shape (T, N, 3)
    velocities: np.ndarray  # shape (T, N, 3)
    masses: np.ndarray  # shape (N,)
    names: list[str]
    energy: np.ndarray  # shape (T,) total energy at each snapshot
    G: float = 1.0  # gravitational constant actually used by the simulator,
    # in whatever unit system this trajectory was generated in. NEVER assume
    # G=1 downstream -- read it from here. For REBOUND with
    # sim.units = ("yr", "AU", "Msun"), G is 4*pi**2 (~39.48), NOT 1.
    body_specs: list[BodySpec] = field(default_factory=list)

    @property
    def n_bodies(self) -> int:
        return len(self.names)

    def relative_energy_error(self) -> np.ndarray:
        """(E(t) - E(0)) / |E(0)|  — the key stability diagnostic (§2.3)."""
        e0 = self.energy[0]
        return (self.energy - e0) / abs(e0)

    def save(self, path: str) -> None:
        np.savez(
            path,
            times=self.times,
            positions=self.positions,
            velocities=self.velocities,
            masses=self.masses,
            names=np.array(self.names),
            energy=self.energy,
            G=np.array(self.G),
        )


def build_simulation(
    bodies: list[BodySpec], seed: int | None = None
) -> rebound.Simulation:
    """
    Construct a REBOUND Simulation from a list of BodySpecs.

    The first body with a=None is treated as the central mass; every other
    body is added as an orbiting particle around the current total mass of
    the simulation so far (REBOUND's standard convention).
    """
    if seed is not None:
        np.random.seed(seed)

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")  # sets G implicitly and consistently
    sim.integrator = "whfast"  # symplectic, kick-drift-kick (§2.4)
    sim.dt = 0.005  # years; small relative to inner orbits

    central = [b for b in bodies if b.a is None]
    if len(central) != 1:
        raise ValueError("Exactly one BodySpec must have a=None (the central body).")
    orbiting = [b for b in bodies if b.a is not None]

    sim.add(m=central[0].mass)  # central body at the origin
    for b in orbiting:
        sim.add(m=b.mass, a=b.a, e=b.e, inc=b.inc)

    sim.move_to_com()  # integrate about the center of mass, not the star
    return sim


def compute_total_energy(sim: rebound.Simulation) -> float:
    """Kinetic + potential energy of the current simulation state (§2.3)."""
    return sim.energy()


def run_simulation(
    bodies: list[BodySpec],
    t_max: float,
    n_snapshots: int,
    seed: int | None = None,
) -> SimResult:
    """
    Integrate `bodies` forward from t=0 to t=t_max, recording n_snapshots
    evenly-spaced states (position, velocity, energy).

    t_max : total integration time, in years
    n_snapshots : number of recorded states (including t=0)
    """
    sim = build_simulation(bodies, seed=seed)
    names = [b.name for b in bodies]
    n = len(names)
    G = sim.G  # read the REAL constant REBOUND is using for this unit system --
    # for ("yr","AU","Msun") this is 4*pi**2 (~39.48), NOT 1. Never hardcode it.

    times = np.linspace(0.0, t_max, n_snapshots)
    positions = np.zeros((n_snapshots, n, 3))
    velocities = np.zeros((n_snapshots, n, 3))
    masses = np.array([p.m for p in sim.particles])
    energy = np.zeros(n_snapshots)

    for i, t in enumerate(times):
        sim.integrate(t)
        for j, p in enumerate(sim.particles):
            positions[i, j] = (p.x, p.y, p.z)
            velocities[i, j] = (p.vx, p.vy, p.vz)
        energy[i] = compute_total_energy(sim)

    return SimResult(
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        names=names,
        energy=energy,
        G=G,
        body_specs=bodies,
    )


# ---------------------------------------------------------------------------
# Milestone system: 1 star + 2 planets, roughly Sun/Earth/Jupiter-like.
# This is deliberately small and deliberately not random — the point of
# Phase 1 is to build trust, so we start with a system whose behavior we can
# sanity-check by eye (near-circular, non-crossing, stable orbits).
# ---------------------------------------------------------------------------


def milestone_system() -> list[BodySpec]:
    return [
        BodySpec(name="star", mass=1.0),  # 1 solar mass
        BodySpec(name="planet_1", mass=3.0e-6, a=1.0, e=0.02),  # Earth-like
        BodySpec(name="planet_2", mass=9.5e-4, a=5.2, e=0.05),  # Jupiter-like
    ]


def run_milestone(
    t_max: float = 20.0, n_snapshots: int = 2000, seed: int = 0
) -> SimResult:
    """Run the Phase 1 milestone system and return the trajectory."""
    return run_simulation(
        milestone_system(), t_max=t_max, n_snapshots=n_snapshots, seed=seed
    )


if __name__ == "__main__":
    result = run_milestone()
    rel_err = result.relative_energy_error()
    print(f"Bodies: {result.names}")
    print(f"Snapshots: {len(result.times)}, t_max={result.times[-1]:.2f} yr")
    print(f"Max |relative energy error| over run: {np.max(np.abs(rel_err)):.3e}")
    result.save("data/simulated/milestone_trajectory.npz")
    print("Saved trajectory to data/simulated/milestone_trajectory.npz")




    
