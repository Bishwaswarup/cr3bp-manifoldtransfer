"""Tests for energy-matched L1/L2 connections."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dynamics import MU_EARTH_MOON as MU, jacobi_constant  # noqa: E402
from src.connections import best_patch, manifold_section_points  # noqa: E402
from src.orbits import differential_correction, richardson_guess  # noqa: E402


@pytest.fixture(scope="module")
def l1():
    s, th = differential_correction(richardson_guess(0.05, "L1", MU)[0], mu=MU)
    return s, 2 * th


def test_section_points_lie_on_section_and_energy(l1):
    s, T = l1
    pts = manifold_section_points(s, T, MU, branch="unstable", n_samples=12)
    assert len(pts) > 6
    assert np.allclose(pts[:, 0], 1 - MU, atol=1e-10)
    C0 = jacobi_constant(s, MU)
    for p in pts:
        assert abs(jacobi_constant(p[:6], MU) - C0) < 1e-6   # ε = 1e-6 perturbation


def test_best_patch_prefers_small_dv():
    dep = np.array([[0.98, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.98, 0.01, 0.0, 0.1, 0.5, 0.0, 0.5]])
    arr = np.array([[0.98, 0.0005, 0.0, 0.1, 0.4, 0.0, 0.0],
                    [0.98, 0.0101, 0.0, 0.1, 0.5, 0.0, 0.5]])
    p = best_patch(dep, arr, max_gap_km=1000)
    assert p["within_gap"]
    assert p["dv_km_s"] < 1e-2                             # picks the matching pair
    assert p["gap_km"] < 50


def test_best_patch_ignores_negative_vx():
    dep = np.array([[0.98, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]])
    arr = np.array([[0.98, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
    assert best_patch(dep, arr) is None
