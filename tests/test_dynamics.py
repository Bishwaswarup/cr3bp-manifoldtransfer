"""
test_dynamics.py
================
Sanity checks for the CR3BP dynamics module.

Run with: pytest tests/test_dynamics.py -v
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dynamics import (
    MU_EARTH_MOON,
    cr3bp_eom,
    jacobi_constant,
    propagate,
    propagate_stm,
    lagrange_points,
    distances,
)


class TestDistances:
    def test_primary_positions(self):
        """At the Earth position (−μ, 0, 0), r1 should be 0."""
        mu = MU_EARTH_MOON
        state_at_earth = np.array([-mu, 0.0, 0.0, 0.0, 0.0, 0.0])
        r1, r2 = distances(state_at_earth, mu)
        assert r1 == pytest.approx(0.0, abs=1e-15)
        assert r2 == pytest.approx(1.0, abs=1e-12)

    def test_moon_position(self):
        """At the Moon position (1−μ, 0, 0), r2 should be 0."""
        mu = MU_EARTH_MOON
        state_at_moon = np.array([1.0 - mu, 0.0, 0.0, 0.0, 0.0, 0.0])
        r1, r2 = distances(state_at_moon, mu)
        assert r2 == pytest.approx(0.0, abs=1e-15)
        assert r1 == pytest.approx(1.0, abs=1e-12)


class TestJacobiConservation:
    def test_jacobi_conserved_along_trajectory(self):
        """Jacobi constant should be conserved to machine precision."""
        mu = MU_EARTH_MOON
        # Start near L1 with a gentle push
        lpts = lagrange_points(mu)
        state0 = lpts["L1"].copy()
        state0[4] = 0.05  # add some vy

        C0 = jacobi_constant(state0, mu)
        t_eval = np.linspace(0.0, 2.0, 200)
        sol = propagate(state0, (0.0, 2.0), mu=mu, t_eval=t_eval)

        C_all = np.array([jacobi_constant(sol.y[:, i], mu) for i in range(sol.y.shape[1])])
        assert np.max(np.abs(C_all - C0)) < 1e-9, (
            f"Jacobi constant drifted by {np.max(np.abs(C_all - C0)):.2e}"
        )

    def test_jacobi_at_lagrange_points(self):
        """Known Jacobi constants at the collinear Lagrange points (approximate)."""
        mu = MU_EARTH_MOON
        lpts = lagrange_points(mu)
        # L1 C value is approximately 3.2 for the Earth–Moon system
        C_L1 = jacobi_constant(lpts["L1"], mu)
        assert 3.0 < C_L1 < 3.5, f"C(L1) = {C_L1:.4f}, expected ~3.1–3.3"


class TestSTM:
    def test_stm_identity_at_t0(self):
        """STM at t=0 should be the identity."""
        mu = MU_EARTH_MOON
        lpts = lagrange_points(mu)
        state0 = lpts["L1"].copy()
        state0[4] = 0.02
        _, stm = propagate_stm(state0, (0.0, 1e-8), mu)
        assert np.allclose(stm, np.eye(6), atol=1e-5)

    def test_stm_volume_preserving(self):
        """The STM determinant should be 1 (Liouville's theorem for Hamiltonian systems).

        Note: the symplectic condition M J Mᵀ = J holds in CANONICAL coordinates
        (px = ẋ − y, py = ẏ + x), NOT in the rotating-frame velocity coords used
        here.  The correct invariant to test is det(Φ) = 1.
        """
        mu = MU_EARTH_MOON
        lpts = lagrange_points(mu)
        state0 = lpts["L1"].copy()
        state0[4] = 0.03  # arbitrary trajectory
        _, stm = propagate_stm(state0, (0.0, 1.0), mu)

        det = np.linalg.det(stm)
        assert abs(det - 1.0) < 1e-6, (
            f"STM determinant = {det:.8f}, expected 1.0 (Liouville's theorem)"
        )


class TestLagrangePoints:
    def test_five_points_returned(self):
        lpts = lagrange_points(MU_EARTH_MOON)
        assert set(lpts.keys()) == {"L1", "L2", "L3", "L4", "L5"}

    def test_l1_between_primaries(self):
        """L1 should be between Earth (x = −μ) and Moon (x = 1−μ)."""
        mu = MU_EARTH_MOON
        lpts = lagrange_points(mu)
        x_L1 = lpts["L1"][0]
        assert -mu < x_L1 < 1.0 - mu

    def test_l4_l5_equilateral(self):
        """L4 and L5 form equilateral triangles with the primaries."""
        mu = MU_EARTH_MOON
        lpts = lagrange_points(mu)
        for key in ("L4", "L5"):
            state = lpts[key]
            r1, r2 = distances(state, mu)
            assert r1 == pytest.approx(1.0, rel=1e-8), f"{key}: r1 = {r1}"
            assert r2 == pytest.approx(1.0, rel=1e-8), f"{key}: r2 = {r2}"

    def test_lagrange_points_are_equilibria(self):
        """Acceleration should be ~0 at all Lagrange points."""
        mu = MU_EARTH_MOON
        lpts = lagrange_points(mu)
        for name, state in lpts.items():
            deriv = cr3bp_eom(0.0, state, mu)
            acc = deriv[3:]  # acceleration components
            assert np.linalg.norm(acc) < 1e-8, (
                f"{name}: |acc| = {np.linalg.norm(acc):.2e} ≠ 0"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
