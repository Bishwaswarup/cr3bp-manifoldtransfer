"""
connections.py
==============
Energy-matched L1 → L2 connections through the lunar region.

An unstable-manifold trajectory leaving an L1 halo towards the Moon and a
stable-manifold trajectory arriving at an L2 halo from the Moon side can
only form a ballistic (zero-ΔV) heteroclinic connection if the two halos
have the same Jacobi constant C. This module

  1. corrects an L1 halo and an L2 halo to the same C
     (:func:`orbits.orbit_at_jacobi`),
  2. propagates the Moon-side unstable manifold of the L1 orbit forward and
     the Moon-side stable manifold of the L2 orbit backward to the section
     x = 1 − μ (the plane through the Moon perpendicular to the Earth–Moon
     line), with exact event detection, and
  3. finds the pair of section points closest in position (y, z) and
     reports the gap and the velocity mismatch (the ΔV of a patch).

Because the two orbits are different, a manifold can never be matched with
itself (the self-intersection pitfall of matching W^u and W^s of a single
orbit).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import L_STAR, MU_EARTH_MOON, V_STAR, cr3bp_eom, cr3bp_eom_stm, jacobi_constant
from .manifolds import stable_unstable_eigenvectors

MOON_RADIUS_KM = 1737.4


def manifold_section_points(
    state0: np.ndarray,
    T: float,
    mu: float = MU_EARTH_MOON,
    branch: str = "unstable",
    side: str = "moon",
    n_samples: int = 100,
    eps: float = 1e-6,
    t_max: float | None = None,
    x_section: float | None = None,
    min_moon_alt_km: float = 100.0,
    return_arcs: bool = False,
):
    """First crossing of x = x_section by each manifold trajectory.

    Parameters
    ----------
    branch : 'unstable' (integrate forward) or 'stable' (integrate backward)
    side   : 'moon' picks the branch whose initial perturbation points
             towards the Moon (+x for an L1 orbit, −x for an L2 orbit)
    min_moon_alt_km : trajectories that pass closer than this to the lunar
             surface before reaching the section are discarded

    Returns
    -------
    (N, 7) array: columns x, y, z, vx, vy, vz, tau (orbit phase in [0, 1));
    with ``return_arcs=True`` also a list of (3, n) position arcs from the
    orbit to the section (for plotting).
    """
    if x_section is None:
        x_section = 1.0 - mu
    if t_max is None:
        t_max = 4.0 * T
    moon = np.array([1.0 - mu, 0.0, 0.0])
    r_min = (MOON_RADIUS_KM + min_moon_alt_km) / L_STAR

    t_s = np.linspace(0.0, T, n_samples, endpoint=False)
    y0 = np.concatenate([state0, np.eye(6).ravel()])
    orb = solve_ivp(cr3bp_eom_stm, (0.0, T), y0, args=(mu,), method="DOP853",
                    rtol=1e-12, atol=1e-12, t_eval=t_s, dense_output=True)
    M = orb.sol(T)[6:].reshape(6, 6)
    v_u, v_s = stable_unstable_eigenvectors(M)
    v0 = v_u if branch == "unstable" else v_s

    towards_moon = np.sign(1.0 - mu - state0[0])          # +1 for L1, −1 for L2
    sgn = 1.0 if np.sign(v0[0]) == towards_moon else -1.0
    if side != "moon":
        sgn = -sgn
    t_dir = 1.0 if branch == "unstable" else -1.0

    def ev_x(t, s, mu): return s[0] - x_section
    ev_x.terminal = True

    def ev_moon(t, s, mu): return np.linalg.norm(s[:3] - moon) - r_min
    ev_moon.terminal, ev_moon.direction = True, -1

    pts, arcs = [], []
    for k, t_k in enumerate(orb.t):
        s_k = orb.y[:6, k]
        v_k = orb.y[6:, k].reshape(6, 6) @ v0
        v_k /= np.linalg.norm(v_k)
        sol = solve_ivp(cr3bp_eom, (0.0, t_dir * t_max), s_k + sgn * eps * v_k, args=(mu,),
                        method="DOP853", rtol=1e-11, atol=1e-12, events=[ev_x, ev_moon],
                        dense_output=return_arcs)
        if sol.t_events[0].size and not sol.t_events[1].size:
            pts.append(np.append(sol.y_events[0][0], t_k / T))
            if return_arcs:
                arcs.append(sol.sol(np.linspace(0.0, sol.t[-1], 400))[:3])
    pts = np.array(pts).reshape(-1, 7)
    return (pts, arcs) if return_arcs else pts


def best_patch(points_dep: np.ndarray, points_arr: np.ndarray, max_gap_km: float = 1000.0) -> dict | None:
    """Best patch between departure and arrival section points.

    Only crossings moving in +x (from the L1 side towards the L2 side) are
    used. Among pairs whose position gap in (y, z) is below ``max_gap_km``
    the pair with the smallest velocity mismatch is returned; the smallest
    gap overall is reported as well.
    """
    dep = points_dep[points_dep[:, 3] > 0] if len(points_dep) else points_dep
    arr = points_arr[points_arr[:, 3] > 0] if len(points_arr) else points_arr
    if len(dep) == 0 or len(arr) == 0:
        return None
    gap = np.linalg.norm(dep[:, None, 1:3] - arr[None, :, 1:3], axis=2) * L_STAR
    dv = np.linalg.norm(dep[:, None, 3:6] - arr[None, :, 3:6], axis=2) * V_STAR
    ok = gap < max_gap_km
    if ok.any():
        score = np.where(ok, dv, np.inf)
        i, j = np.unravel_index(np.argmin(score), score.shape)
    else:
        i, j = np.unravel_index(np.argmin(gap), gap.shape)
    return {
        "state_dep": dep[i, :6],
        "state_arr": arr[j, :6],
        "tau_dep": dep[i, 6],
        "tau_arr": arr[j, 6],
        "gap_km": float(gap[i, j]),
        "dv_km_s": float(dv[i, j]),
        "min_gap_km": float(gap.min()),
        "dv_at_min_gap_km_s": float(dv.flat[np.argmin(gap)]),
        "within_gap": bool(ok.any()),
    }


def energy_matched_connection(
    C: float,
    family_L1: list[dict],
    family_L2: list[dict],
    mu: float = MU_EARTH_MOON,
    n_samples: int = 100,
    max_gap_km: float = 1000.0,
) -> dict:
    """Build the L1/L2 halo pair at Jacobi constant C and search for a patch."""
    from .orbits import orbit_at_jacobi, z_amplitude

    s1, th1 = orbit_at_jacobi(family_L1, C, mu)
    s2, th2 = orbit_at_jacobi(family_L2, C, mu)
    dep = manifold_section_points(s1, 2 * th1, mu, branch="unstable", n_samples=n_samples)
    arr = manifold_section_points(s2, 2 * th2, mu, branch="stable", n_samples=n_samples)
    patch = best_patch(dep, arr, max_gap_km)
    return {
        "C": C,
        "C_check": (jacobi_constant(s1, mu), jacobi_constant(s2, mu)),
        "orbit_L1": (s1, 2 * th1),
        "orbit_L2": (s2, 2 * th2),
        "Az_L1_km": z_amplitude(s1, 2 * th1, mu) * L_STAR,
        "Az_L2_km": z_amplitude(s2, 2 * th2, mu) * L_STAR,
        "section_dep": dep,
        "section_arr": arr,
        "patch": patch,
    }
