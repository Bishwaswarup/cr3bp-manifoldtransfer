"""Tests for periodic-orbit computation (Richardson guess, correction, continuation)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dynamics import MU_EARTH_MOON as MU, T_STAR, jacobi_constant, lagrange_points, propagate  # noqa: E402
from src.orbits import (compute_halo_family, correct_to_jacobi, differential_correction,  # noqa: E402
                        orbit_at_jacobi, periodicity_error, richardson_guess, stability_index)

DAYS = T_STAR / 86400.0


@pytest.fixture(scope="module")
def l1_halo():
    s, th = differential_correction(richardson_guess(0.05, "L1", MU)[0], mu=MU)
    return s, 2 * th


@pytest.mark.parametrize("lp", ["L1", "L2"])
@pytest.mark.parametrize("Az", [0.01, 0.05])
def test_corrected_orbit_is_periodic(lp, Az):
    s, th = differential_correction(richardson_guess(Az, lp, MU)[0], mu=MU)
    assert periodicity_error(s, 2 * th, MU) < 1e-8


@pytest.mark.parametrize("lp", ["L1", "L2"])
def test_richardson_guess_is_close(lp):
    g, th_g = richardson_guess(0.05, lp, MU)
    s, th = differential_correction(g, mu=MU)
    # third-order guess: within ~1000 km in position and 2 % in half-period
    assert np.linalg.norm(g[:3] - s[:3]) * 384_400 < 1000
    assert abs(th_g - th) / th < 0.02


def test_halo_is_genuinely_three_dimensional(l1_halo):
    s, T = l1_halo
    sol = propagate(s, (0, T), mu=MU, t_eval=np.linspace(0, T, 400))
    assert np.max(np.abs(sol.y[2])) > 0.04            # did not collapse to z = 0
    assert np.max(np.abs(sol.y[2])) < 0.07


def test_known_l1_halo_period(l1_halo):
    _, T = l1_halo
    assert 11.5 < T * DAYS < 12.5                      # Earth–Moon L1 halos: ~12 d


def test_orbit_stays_near_l1(l1_halo):
    s, T = l1_halo
    sol = propagate(s, (0, T), mu=MU)
    xL1 = lagrange_points(MU)["L1"][0]
    assert np.all(np.abs(sol.y[0] - xL1) < 0.05)


def test_jacobi_constant_conserved_on_orbit(l1_halo):
    s, T = l1_halo
    sol = propagate(s, (0, T), mu=MU, t_eval=np.linspace(0, T, 50))
    C = [jacobi_constant(sol.y[:, k], MU) for k in range(50)]
    assert np.ptp(C) < 1e-10


def test_unstable_small_halo(l1_halo):
    s, T = l1_halo
    assert stability_index(s, T, MU) > 100


def test_family_is_continuous_and_periodic():
    fam = compute_halo_family(0.01, 0.06, n_orbits=6, lp="L1", ds=0.005)
    assert len(fam) >= 4
    for m in fam:
        assert periodicity_error(m["state0"], m["T"], MU) < 1e-8
    Az = [m["Az"] for m in fam]
    assert all(b > a for a, b in zip(Az, Az[1:]))          # amplitude grows monotonically
    C = [m["C"] for m in fam]
    assert all(b < a for a, b in zip(C, C[1:]))           # energy rises along the family


def test_correct_to_jacobi(l1_halo):
    s, _ = l1_halo
    C0 = jacobi_constant(s, MU)
    s2, th2 = correct_to_jacobi(s, C0 - 0.002, MU)
    assert abs(jacobi_constant(s2, MU) - (C0 - 0.002)) < 1e-10
    assert periodicity_error(s2, 2 * th2, MU) < 1e-8


def test_orbit_at_jacobi_out_of_range():
    fam = compute_halo_family(0.01, 0.03, n_orbits=3, lp="L1", ds=0.005)
    with pytest.raises(ValueError):
        orbit_at_jacobi(fam, 2.5, MU)
