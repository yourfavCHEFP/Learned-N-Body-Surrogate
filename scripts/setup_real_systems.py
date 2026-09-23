"""
scripts/setup_real_systems.py

Converts NASA Exoplanet Archive data into REBOUND simulations for
# known multi-planet systems (TRAPPIST-1, Kepler-90, etc.).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rebound

DATA_DIR = Path("data/raw")
SIMULATED_DIR = Path("data/simulated")
SIMULATED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SYSTEMS = [
    "TRAPPIST-1",
    "Kepler-90",
    "HD 10180",
    "Kepler-11",
    "HR 8799",
]


def setup_system_simulation(system_name, df, duration_years=20, timestep=0.01):
    # Match hostnames case-insensitively / stripped
    mask = (
        df["hostname"].astype(str).str.strip().str.lower()
        == system_name.strip().lower()
    )
    system_planets = df[mask].copy()

    if len(system_planets) == 0:
        print(f"No planets found for system {system_name}")
        return None, None

    system_planets = system_planets.sort_values("pl_orbsmax")
    star_mass = float(system_planets.iloc[0]["st_mass"])

    print(f"\nSetting up {system_name}:")
    print(f"  Host star mass: {star_mass:.3f} M☉")
    print(f"  Planets found: {len(system_planets)}")

    sim = rebound.Simulation()
    sim.units = ("AU", "Msun", "yr")
    sim.integrator = "whfast"
    sim.dt = timestep
    sim.add(m=star_mass)

    for _, planet in system_planets.iterrows():
        name = planet["pl_name"]
        a = float(planet["pl_orbsmax"])

        # Calculate mass in solar masses:
        if pd.notna(planet["pl_bmassj"]):
            m_solar = float(planet["pl_bmassj"]) * 0.000954588
        elif pd.notna(planet["pl_bmasse"]):
            m_solar = float(planet["pl_bmasse"]) * 3.003e-6
        else:
            continue

        e = float(planet["pl_orbeccen"]) if pd.notna(planet["pl_orbeccen"]) else 0.0
        inc = (
            np.deg2rad(float(planet["pl_orbincl"]))
            if pd.notna(planet["pl_orbincl"])
            else 0.0
        )
        f = np.random.uniform(0, 2 * np.pi)

        print(f"  Adding {name}: a={a:.4f} AU, m={m_solar:.6e} M☉, e={e:.3f}")
        try:
            sim.add(m=m_solar, a=a, e=e, inc=inc, f=f)
        except Exception as ex:
            print(f"    Warning: Could not add {name}: {ex}")

    sim.move_to_com()
    return sim, system_planets


def run_and_save(sim, system_name, duration_years=20, timestep=0.01):
    n_steps = int(duration_years / timestep)
    times = np.linspace(0, duration_years, n_steps)
    n_bodies = sim.N

    positions = np.zeros((n_steps, n_bodies, 3))
    velocities = np.zeros((n_steps, n_bodies, 3))

    for i, t in enumerate(times):
        sim.integrate(t)
        for j, p in enumerate(sim.particles):
            positions[i, j] = [p.x, p.y, p.z]
            velocities[i, j] = [p.vx, p.vy, p.vz]

    masses = np.array([p.m for p in sim.particles])
    names = [f"{system_name}_star"] + [
        f"{system_name}_p{k}" for k in range(1, n_bodies)
    ]

    clean_name = system_name.lower().replace(" ", "_").replace("-", "_")
    save_path = SIMULATED_DIR / f"{clean_name}_trajectory.npz"

    np.savez(
        save_path,
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        names=names,
    )
    print(f"✅ Saved trajectory to {save_path}")
    return save_path


def main():
    csv_path = DATA_DIR / "exoplanets_raw.csv"
    if not csv_path.exists():
        print("Please run scripts/fetch_nasa_exoplanets.py first!")
        return

    df = pd.read_csv(csv_path)
    for name in TARGET_SYSTEMS:
        sim, p_df = setup_system_simulation(name, df)
        if sim and sim.N > 1:
            run_and_save(sim, name)


if __name__ == "__main__":
    main()
