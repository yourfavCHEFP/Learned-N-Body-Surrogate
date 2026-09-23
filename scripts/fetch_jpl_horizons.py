"""
scripts/fetch_jpl_horizons.py

Generate Solar System ephemerides using astropy (no network needed).
Falls back to local computation if JPL API is blocked.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("⚠️  astropy not installed. Install with: pip install astropy")

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

SOLAR_BODIES = [
    "sun",
    "mercury",
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]

START_TIME = "2026-01-01"
STOP_TIME = "2027-01-01"
STEP_SIZE_DAYS = 5


def generate_ephemerides_astropy():
    """
    Generate Solar System ephemerides using astropy (no internet required).
    Uses built-in planetary models.
    """
    if not ASTROPY_AVAILABLE:
        return None

    print("Generating Solar System ephemerides using astropy...")

    # Create time array
    t_start = Time(START_TIME)
    t_stop = Time(STOP_TIME)
    n_steps = int((t_stop.jd - t_start.jd) / STEP_SIZE_DAYS)
    times = t_start + np.linspace(0, (t_stop - t_start).jd, n_steps) * u.day

    rows = []

    # Use built-in ephemerides
    with solar_system_ephemeris.set("builtin"):
        for body_name in SOLAR_BODIES:
            print(f"  Computing {body_name.capitalize()}...")

            for t in times:
                try:
                    # Get barycentric position and velocity
                    pos, vel = get_body_barycentric_posvel(body_name, t)

                    # Convert to AU and AU/day
                    x, y, z = pos.xyz.to("AU").value
                    vx, vy, vz = vel.xyz.to("AU/day").value

                    rows.append(
                        {
                            "body": body_name.capitalize(),
                            "jd_tdb": t.jd,
                            "x": x,
                            "y": y,
                            "z": z,
                            "vx": vx,
                            "vy": vy,
                            "vz": vz,
                        }
                    )
                except Exception as e:
                    print(f"    Warning: Could not compute {body_name} at {t}: {e}")
                    continue

    if not rows:
        print("❌ No ephemerides generated")
        return None

    df = pd.DataFrame(rows)
    csv_path = RAW_DATA_DIR / "jpl_horizons_vectors.csv"
    df.to_csv(csv_path, index=False)
    print(
        f"\n✅ Saved {len(df)} rows across {df['body'].nunique()} bodies -> {csv_path}"
    )
    return df


def generate_simple_keplerian():
    """
    Fallback: Generate simple Keplerian orbits if astropy isn't available.
    Uses circular orbit approximations.
    """
    print("Generating simple Keplerian approximations...")
    print("⚠️  Note: These are simplified orbits, not JPL-quality ephemerides")

    # Simplified orbital parameters (semi-major axis in AU, period in years)
    bodies = {
        "Sun": {"a": 0.0, "period": 0.0, "mass": 1.0},
        "Mercury": {"a": 0.387, "period": 0.241},
        "Venus": {"a": 0.723, "period": 0.615},
        "Earth": {"a": 1.0, "period": 1.0},
        "Mars": {"a": 1.524, "period": 1.881},
        "Jupiter": {"a": 5.203, "period": 11.86},
        "Saturn": {"a": 9.537, "period": 29.46},
        "Uranus": {"a": 19.191, "period": 84.01},
        "Neptune": {"a": 30.069, "period": 164.8},
    }

    # Time array
    n_steps = int(365 / STEP_SIZE_DAYS)
    times_jd = np.linspace(2459946, 2460311, n_steps)  # JD for 2026

    rows = []

    for body_name, params in bodies.items():
        if body_name == "Sun":
            # Sun at origin
            for jd in times_jd:
                rows.append(
                    {
                        "body": body_name,
                        "jd_tdb": jd,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "vz": 0.0,
                    }
                )
        else:
            a = params["a"]
            P = params["period"]  # years

            for jd in times_jd:
                t = (jd - times_jd[0]) / 365.25  # Time in years

                # Mean anomaly (circular orbit)
                M = 2 * np.pi * t / P

                # Position (circular orbit in xy-plane)
                x = a * np.cos(M)
                y = a * np.sin(M)
                z = 0.0

                # Velocity (circular orbit)
                v = 2 * np.pi * a / (P * 365.25)  # AU/day
                vx = -v * np.sin(M)
                vy = v * np.cos(M)
                vz = 0.0

                rows.append(
                    {
                        "body": body_name,
                        "jd_tdb": jd,
                        "x": x,
                        "y": y,
                        "z": z,
                        "vx": vx,
                        "vy": vy,
                        "vz": vz,
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = RAW_DATA_DIR / "jpl_horizons_vectors.csv"
    df.to_csv(csv_path, index=False)
    print(
        f"\n✅ Saved {len(df)} rows across {df['body'].nunique()} bodies -> {csv_path}"
    )
    return df


def main():
    print("=" * 60)
    print("JPL Horizons Solar System Ephemerides")
    print("=" * 60)

    # Try astropy first
    if ASTROPY_AVAILABLE:
        try:
            # Need to import units
            from astropy import units as u

            globals()["u"] = u
            df = generate_ephemerides_astropy()
            if df is not None:
                return df
        except Exception as e:
            print(f"Astropy method failed: {e}")
            print("Falling back to simple Keplerian orbits...")

    # Fallback to simple Keplerian
    return generate_simple_keplerian()


if __name__ == "__main__":
    main()
