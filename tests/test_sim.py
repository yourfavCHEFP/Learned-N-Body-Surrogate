# This is a simple test file for the simulation code

"""
Phase 1 sanity tests.

These don't test ML — they test that we can trust REBOUND's output before
building anything on top of it. If these fail, nothing downstream matters.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody_surrogate.sim import (
    BodySpec,
    run_milestone,
    run_simulation,
)


def test_energy_conservation():
    """WHFast is symplectic: relative energy error should stay tiny and bounded."""
    result = run_milestone(t_max=20.0, n_snapshots=500)
    rel_err = result.relative_energy_error()
    assert (
        np.max(np.abs(rel_err)) < 1e-6
    ), "Energy drifted more than expected for a symplectic integrator"


def test_orbits_stay_bound():
    """Planets shouldn't fly off to infinity or crash into the star over a short, stable run."""
    result = run_milestone(t_max=20.0, n_snapshots=500)
    star_pos = result.positions[:, 0, :]
    for j, name in enumerate(result.names[1:], start=1):
        dist = np.linalg.norm(result.positions[:, j, :] - star_pos, axis=1)
        assert (
            dist.min() > 0.01
        ), f"{name} got implausibly close to the star (possible collision)"
        assert dist.max() < 50.0, f"{name} escaped to an implausible distance"


def test_center_of_mass_is_stationary():
    """We integrate about the center of mass, so it should not drift."""
    result = run_milestone(t_max=20.0, n_snapshots=200)
    com = np.average(result.positions, axis=1, weights=result.masses)
    com_drift = np.linalg.norm(com - com[0], axis=1)
    assert com_drift.max() < 1e-6, "Center of mass drifted — check move_to_com()"


def test_two_body_matches_kepler_period():
    """
    For a simple 2-body system, Kepler's third law gives an exact expected
    period: T^2 = a^3 / (M_total) in these units (G=1). Use this as an
    independent check that REBOUND is doing what we think it's doing.
    """
    bodies = [
        BodySpec(name="star", mass=1.0),
        BodySpec(name="planet", mass=1e-6, a=1.0, e=0.0),
    ]
    result = run_simulation(bodies, t_max=2.0, n_snapshots=2000)
    star_pos = result.positions[:, 0, :]
    rel_pos = result.positions[:, 1, :] - star_pos
    angle = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
    # find when the planet returns near its starting angle after leaving it
    unwrapped = np.unwrap(angle)
    crossed = np.where(np.abs(unwrapped - unwrapped[0]) >= 2 * np.pi)[0]
    assert len(crossed) > 0, "Planet never completed a full orbit in the given timespan"
    period_est = result.times[crossed[0]]
    expected_period = np.sqrt(1.0**3 / 1.0)  # a=1 AU, M_total ~= 1 Msun => T ~= 1 yr
    assert abs(period_est - expected_period) / expected_period < 0.05


def test_save_and_load_trajectory():
    """Verify that trajectory data can be saved and loaded without corruption."""
    result = run_milestone(t_max=5.0, n_snapshots=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_trajectory.npz"
        result.save(str(path))

        # Verify file exists
        assert path.exists(), "Trajectory file was not created"
        assert path.stat().st_size > 0, "Trajectory file is empty"

        # Load and verify data integrity
        data = np.load(path)
        assert "times" in data, "Missing 'times' in saved data"
        assert "positions" in data, "Missing 'positions' in saved data"
        assert "velocities" in data, "Missing 'velocities' in saved data"
        assert "masses" in data, "Missing 'masses' in saved data"
        assert "energy" in data, "Missing 'energy' in saved data"

        # Verify loaded data matches original
        assert np.allclose(
            data["times"], result.times
        ), "Times mismatch after save/load"
        assert np.allclose(
            data["positions"], result.positions
        ), "Positions mismatch after save/load"
        assert np.allclose(
            data["velocities"], result.velocities
        ), "Velocities mismatch after save/load"
        assert np.allclose(
            data["masses"], result.masses
        ), "Masses mismatch after save/load"
        assert np.allclose(
            data["energy"], result.energy
        ), "Energy mismatch after save/load"


if __name__ == "__main__":
    test_energy_conservation()
    test_orbits_stay_bound()
    test_center_of_mass_is_stationary()
    test_two_body_matches_kepler_period()
    test_save_and_load_trajectory()
    print("All Phase 1 sanity tests passed.")
