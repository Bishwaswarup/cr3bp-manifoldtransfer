"""
manifolds.py
============
Computation of stable and unstable invariant manifolds associated with
periodic orbits (halo/Lyapunov) in the CR3BP.

Algorithm
---------
1. Compute the monodromy matrix Φ(T, 0) by integrating the variational
   equations over one full period.
2. Extract the stable/unstable eigenvectors from Φ.
   - Eigenvalue λ > 1   →  unstable manifold direction (grow forward in time)
   - Eigenvalue 0 < λ < 1 →  stable manifold direction (grow backward in time)
3. Perturb the orbit along these eigenvectors at n_manifold points uniformly
   sampled around the orbit.
4. Propagate the perturbed initial conditions backward (stable) or forward
   (unstable) to build the manifold tubes.

References
----------
Parker, J. S., & Anderson, R. L. (2014).
  *Low-Energy Lunar Trajectory Design*. JPL Deep Space Communications and
  Navigation Series.
Koon, W. S. et al. (2008). *Dynamical Systems, the Three-Body Problem and
  Space Mission Design*. Springer.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import (
    MU_EARTH_MOON,
    propagate,
    propagate_stm,
    cr3bp_eom,
)


# ── Monodromy matrix ───────────────────────────────────────────────────────────


def monodromy_matrix(
    state0: np.ndarray,
    T: float,
    mu: float = MU_EARTH_MOON,
) -> np.ndarray:
    """Compute the monodromy matrix Φ(T, 0) for a periodic orbit.

    Parameters
    ----------
    state0 : (6,) initial state on the periodic orbit
    T      : period of the orbit
    mu     : mass parameter

    Returns
    -------
    M : (6, 6) monodromy matrix
    """
    _, stm = propagate_stm(state0, (0.0, T), mu)
    return stm


def stable_unstable_eigenvectors(
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the most unstable and most stable eigenvectors of the monodromy.

    The monodromy matrix of a hyperbolic periodic orbit has eigenvalues
    {λ, 1/λ, 1, 1, e^{iθ}, e^{-iθ}} where λ > 1.

    Parameters
    ----------
    M : (6, 6) monodromy matrix

    Returns
    -------
    v_u : (6,) real unstable eigenvector (eigenvalue > 1)
    v_s : (6,) real stable eigenvector   (eigenvalue ∈ (0,1))
    """
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # Real eigenvalues only
    real_mask = np.abs(eigenvalues.imag) < 1e-6
    real_evals = eigenvalues[real_mask].real
    real_evecs = eigenvectors[:, real_mask].real

    # Most unstable eigenvalue (largest > 1)
    unstable_idx = np.argmax(real_evals)
    v_u = real_evecs[:, unstable_idx]
    v_u = v_u / np.linalg.norm(v_u)

    # Most stable eigenvalue (smallest positive, < 1)
    positive_mask = real_evals > 0
    stable_candidates = real_evals[positive_mask]
    stable_evecs = real_evecs[:, positive_mask]
    # Exclude the unstable one (which is the largest)
    stable_only_mask = stable_candidates < 0.9
    if stable_only_mask.any():
        stable_idx = np.argmin(stable_candidates[stable_only_mask])
        v_s = stable_evecs[:, stable_only_mask][:, stable_idx]
    else:
        # Fallback: smallest eigenvalue
        stable_idx = np.argmin(real_evals)
        v_s = real_evecs[:, stable_idx]
    v_s = v_s / np.linalg.norm(v_s)

    return v_u, v_s


# ── Manifold tubes ─────────────────────────────────────────────────────────────


def _orbit_trajectory(
    state0: np.ndarray,
    T: float,
    mu: float,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate state0 over one period and return (times, states)."""
    t_eval = np.linspace(0.0, T, n_points)
    sol = propagate(state0, (0.0, T), mu=mu, t_eval=t_eval)
    return sol.t, sol.y.T  # shape (n_points, 6)


def compute_manifold_tube(
    state0: np.ndarray,
    T: float,
    mu: float = MU_EARTH_MOON,
    branch: str = "unstable",
    direction: int = +1,
    n_manifold: int = 100,
    eps: float = 1e-6,
    t_manifold: float | None = None,
    n_points: int = 500,
) -> list[np.ndarray]:
    """Compute a bundle of manifold trajectories (the manifold tube).

    Parameters
    ----------
    state0      : (6,) initial state on the periodic orbit (at t=0)
    T           : period of the orbit
    mu          : mass parameter
    branch      : 'unstable' or 'stable'
    direction   : +1 or -1 (two branches of each manifold)
    n_manifold  : number of trajectories in the tube
    eps         : perturbation size along the eigenvector
    t_manifold  : integration time for each manifold arc (default: 3T for
                  unstable, 3T backward for stable)
    n_points    : number of time points per manifold trajectory

    Returns
    -------
    list of (n_points, 6) arrays, one per manifold trajectory
    """
    if t_manifold is None:
        t_manifold = 3.0 * T

    # Sample n_manifold points uniformly around the orbit
    t_sample = np.linspace(0.0, T, n_manifold, endpoint=False)
    sol_orbit = propagate(state0, (0.0, T), mu=mu, t_eval=t_sample)
    orbit_states = sol_orbit.y.T  # (n_manifold, 6)

    # Monodromy and eigenvectors at the fiducial point
    M = monodromy_matrix(state0, T, mu)
    v_u, v_s = stable_unstable_eigenvectors(M)

    # Choose which eigenvector to use
    v_base = v_u if branch == "unstable" else v_s

    # Integration direction: unstable → forward (+), stable → backward (−)
    int_sign = +1.0 if branch == "unstable" else -1.0

    trajectories = []
    for i, t_i in enumerate(t_sample):
        # State on the orbit at time t_i
        s_i = orbit_states[i, :]

        # Map the eigenvector to this point on the orbit using the STM
        _, stm_i = propagate_stm(state0, (0.0, t_i), mu)
        v_i = stm_i @ v_base
        v_i = v_i / np.linalg.norm(v_i)

        # Perturb
        s_pert = s_i + direction * eps * v_i

        # Integrate manifold arc
        t_end = int_sign * t_manifold
        t_eval_m = np.linspace(0.0, t_end, n_points)
        sol_m = propagate(s_pert, (0.0, t_end), mu=mu, t_eval=t_eval_m)
        trajectories.append(sol_m.y.T)  # (n_points, 6)

    return trajectories


def compute_both_manifolds(
    state0: np.ndarray,
    T: float,
    mu: float = MU_EARTH_MOON,
    n_manifold: int = 80,
    eps: float = 1e-6,
    t_manifold: float | None = None,
    n_points: int = 500,
) -> dict[str, list[np.ndarray]]:
    """Convenience wrapper: compute all four half-manifolds.

    Returns a dict with keys:
        'unstable_pos', 'unstable_neg', 'stable_pos', 'stable_neg'
    Each value is a list of (n_points, 6) trajectory arrays.
    """
    kw = dict(mu=mu, n_manifold=n_manifold, eps=eps,
              t_manifold=t_manifold, n_points=n_points)
    return {
        "unstable_pos": compute_manifold_tube(state0, T, branch="unstable", direction=+1, **kw),
        "unstable_neg": compute_manifold_tube(state0, T, branch="unstable", direction=-1, **kw),
        "stable_pos":   compute_manifold_tube(state0, T, branch="stable",   direction=+1, **kw),
        "stable_neg":   compute_manifold_tube(state0, T, branch="stable",   direction=-1, **kw),
    }
