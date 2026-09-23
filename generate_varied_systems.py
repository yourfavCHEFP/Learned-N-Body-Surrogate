"""
generate_varied_systems.py
Generates diverse N-body systems for cross-system generalization testing.
"""

from pathlib import Path

import numpy as np
import rebound

SIMULATED_DIR = Path("data/simulated")
SIMULATED_DIR.mkdir(parents=True, exist_ok=True)


def generate_4body_system():
    """Generate 4-body system (1 star + 3 planets)."""
    sim = rebound.Simulation()
    sim.units = ("AU", "Msun", "yr")
    sim.integrator = "whfast"
    sim.dt = 0.01

    # Star
    sim.add(m=1.0)

    # 3 planets
    sim.add(m=3e-6, a=0.7, e=0.05, f=np.random.uniform(0, 2 * np.pi))
    sim.add(m=3e-6, a=1.3, e=0.02, f=np.random.uniform(0, 2 * np.pi))
    sim.add(m=9.5e-4, a=3.8, e=0.03, f=np.random.uniform(0, 2 * np.pi))

    sim.move_to_com()
    return sim


def generate_mdwarf_system():
    """Generate M-dwarf system (0.3 Msun star)."""
    sim = rebound.Simulation()
    sim.units = ("AU", "Msun", "yr")
    sim.integrator = "whfast"
    sim.dt = 0.01

    # M-dwarf star
    sim.add(m=0.3)

    # 2 close-in planets
    sim.add(m=2e-6, a=0.1, e=0.01, f=np.random.uniform(0, 2 * np.pi))
    sim.add(m=4e-6, a=0.2, e=0.02, f=np.random.uniform(0, 2 * np.pi))

    sim.move_to_com()
    return sim


def generate_eccentric_system():
    """Generate system with high eccentricity orbits."""
    sim = rebound.Simulation()
    sim.units = ("AU", "Msun", "yr")
    sim.integrator = "whfast"
    sim.dt = 0.01

    # Star
    sim.add(m=1.0)

    # Eccentric planets
    sim.add(m=3e-6, a=1.0, e=0.3, f=np.random.uniform(0, 2 * np.pi))
    sim.add(m=9.5e-4, a=5.2, e=0.5, f=np.random.uniform(0, 2 * np.pi))

    sim.move_to_com()
    return sim


def run_and_save(sim, name, duration=20, timestep=0.01):
    """Run simulation and save trajectory."""
    n_steps = int(duration / timestep)
    times = np.linspace(0, duration, n_steps)
    n_bodies = sim.N

    positions = np.zeros((n_steps, n_bodies, 3))
    velocities = np.zeros((n_steps, n_bodies, 3))

    for i, t in enumerate(times):
        sim.integrate(t)
        for j, p in enumerate(sim.particles):
            positions[i, j] = [p.x, p.y, p.z]
            velocities[i, j] = [p.vx, p.vy, p.vz]

    masses = np.array([p.m for p in sim.particles])
    names_list = [f"{name}_body{i}" for i in range(n_bodies)]

    save_path = SIMULATED_DIR / f"{name}_trajectory.npz"
    np.savez(
        save_path,
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        names=names_list,
    )
    print(f"✅ Saved {name} to {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("Generating Varied Systems for Generalization Testing")
    print("=" * 60)

    # Generate test systems
    print("\n1. 4-body system (3 planets)...")
    sim = generate_4body_system()
    run_and_save(sim, "test_4body")

    print("\n2. M-dwarf system (0.3 Msun)...")
    sim = generate_mdwarf_system()
    run_and_save(sim, "test_mdwarf")

    print("\n3. Eccentric system (e > 0.3)...")
    sim = generate_eccentric_system()
    run_and_save(sim, "test_eccentric")

    print("\n" + "=" * 60)
    print("All test systems generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
