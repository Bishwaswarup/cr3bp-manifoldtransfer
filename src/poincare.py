"""
poincare.py
===========
Poincaré section tools for the CR3BP manifold intersection search.

A Poincaré section records the state of each manifold trajectory each time
it crosses a fixed surface in phase space.  Here we use two common choices:

  * y = 0   (the xz-plane, equivalently the x–z rotating frame plane)
  * x = x_s  (a vertical plane at a chosen x-coordinate)

Crossing detection is done by monitoring sign changes in the section
coordinate during integration.

References
----------
Koon, W. S. et al. (2008). *Dynamical Systems, the Three-Body Problem and
  Space Mission Design*. Springer.  Chapters 4 & 6.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .dynamics import MU_EARTH_MOON, cr3bp_eom, propagate


# ── Section crossing detection ────────────────────────────────────────────────


def _make_y_zero_event(direction: float = 0.0):
    """Create a y=0 event function for solve_ivp."""
    def event(t, s, mu): return s[1]
    event.terminal = False
    event.direction = direction   # 0 = both; +1/-1 = one direction
    return event


def _make_x_section_event(x_section: float, direction: float = 0.0):
    """Create a x = x_section event function for solve_ivp."""
    def event(t, s, mu): return s[0] - x_section
    event.terminal = False
    event.direction = direction
    return event


def poincare_crossings_y0(
    trajectories: list[np.ndarray],
    state0: np.ndarray,
    T_manifold: float,
    mu: float = MU_EARTH_MOON,
    direction: float = 0.0,
    n_cross: int = 1,
) -> np.ndarray:
    """Record Poincaré section states (y=0) for a list of manifold trajectories.

    Instead of re-integrating, we interpolate the pre-computed trajectories
    to find y=0 crossings.

    Parameters
    ----------
    trajectories : list of (n_points, 6) arrays from compute_manifold_tube
    state0       : fiducial initial state on the orbit (unused; for API consistency)
    T_manifold   : integration time used for the manifold trajectories
    mu           : mass parameter
    direction    : +1 = y going from – to +; −1 = y going from + to –; 0 = both
    n_cross      : maximum number of crossings to record per trajectory

    Returns
    -------
    crossings : (N_cross, 6) array of states at the Poincaré section
    """
    crossings = []
    for traj in trajectories:
        y_vals = traj[:, 1]
        count = 0
        for i in range(len(y_vals) - 1):
            y0, y1 = y_vals[i], y_vals[i + 1]
            if y0 * y1 > 0:
                continue  # no sign change
            if direction > 0 and y1 < y0:
                continue
            if direction < 0 and y1 > y0:
                continue
            # Linear interpolation to refine crossing
            alpha = -y0 / (y1 - y0)
            s_cross = traj[i] + alpha * (traj[i + 1] - traj[i])
            crossings.append(s_cross)
            count += 1
            if count >= n_cross:
                break

    if not crossings:
        return np.empty((0, 6))
    return np.vstack(crossings)


def poincare_crossings_x(
    trajectories: list[np.ndarray],
    x_section: float,
    direction: float = 0.0,
    n_cross: int = 1,
) -> np.ndarray:
    """Record Poincaré section states at x = x_section.

    Parameters
    ----------
    trajectories : list of (n_points, 6) arrays
    x_section    : x-coordinate of the section
    direction    : +1 = x increasing; −1 = x decreasing; 0 = both
    n_cross      : max crossings per trajectory

    Returns
    -------
    crossings : (N_cross, 6) array
    """
    crossings = []
    for traj in trajectories:
        x_vals = traj[:, 0] - x_section
        count = 0
        for i in range(len(x_vals) - 1):
            x0, x1 = x_vals[i], x_vals[i + 1]
            if x0 * x1 > 0:
                continue
            if direction > 0 and x1 < x0:
                continue
            if direction < 0 and x1 > x0:
                continue
            alpha = -x0 / (x1 - x0)
            s_cross = traj[i] + alpha * (traj[i + 1] - traj[i])
            crossings.append(s_cross)
            count += 1
            if count >= n_cross:
                break

    if not crossings:
        return np.empty((0, 6))
    return np.vstack(crossings)


# ── Intersection search ────────────────────────────────────────────────────────


def find_manifold_intersections(
    crossings_departure: np.ndarray,
    crossings_arrival: np.ndarray,
    position_only: bool = True,
    n_best: int = 5,
) -> list[dict]:
    """Find near-intersections between two manifold Poincaré sections.

    Caution: the departure and arrival crossings must come from *different*
    orbits. Matching W^u and W^s of one orbit returns its own homoclinic
    self-intersections (or the orbit itself), not a transfer. For L1 → L2
    connections use ``connections.energy_matched_connection``.

    Searches for pairs of section crossing states (one from the unstable
    manifold of the departure orbit, one from the stable manifold of the
    arrival orbit) that are close in position space.

    Parameters
    ----------
    crossings_departure : (N1, 6) array — Poincaré section of unstable manifold
    crossings_arrival   : (N2, 6) array — Poincaré section of stable manifold
    position_only       : if True, measure distance in (x, z) only (the y=0
                          section only constrains position up to y; matching
                          (x, z) is sufficient for a ΔV patch)
    n_best              : number of best intersection candidates to return

    Returns
    -------
    list of dicts with keys:
        'state_dep'   : (6,) state on the departure manifold
        'state_arr'   : (6,) state on the arrival manifold
        'pos_dist'    : position distance at the section
        'delta_v'     : (3,) velocity difference [Δvx, Δvy, Δvz] in nd units
        'dv_norm'     : |Δv| (non-dimensional)
        'dv_km_s'     : |Δv| in km/s
    """
    from .dynamics import V_STAR

    if crossings_departure.shape[0] == 0 or crossings_arrival.shape[0] == 0:
        return []

    candidates = []
    for s_d in crossings_departure:
        for s_a in crossings_arrival:
            if position_only:
                dist = np.sqrt((s_d[0] - s_a[0])**2 + (s_d[2] - s_a[2])**2)
            else:
                dist = np.linalg.norm(s_d[:3] - s_a[:3])

            dv = s_a[3:] - s_d[3:]
            dv_norm = np.linalg.norm(dv)
            candidates.append({
                "state_dep": s_d.copy(),
                "state_arr": s_a.copy(),
                "pos_dist": dist,
                "delta_v": dv.copy(),
                "dv_norm": dv_norm,
                "dv_km_s": dv_norm * V_STAR,
            })

    # Sort by position distance first, then ΔV
    candidates.sort(key=lambda c: (c["pos_dist"], c["dv_norm"]))
    return candidates[:n_best]
