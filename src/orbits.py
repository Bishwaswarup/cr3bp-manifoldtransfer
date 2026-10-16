"""
orbits.py
=========
Periodic orbits about the collinear libration points L1 and L2:

  1. Third-order Richardson (1980) analytic initial guess for halo orbits
  2. Single-shooting differential correction (Howell 1984) that targets the
     perpendicular return to the xz-plane, y(T/2) = vx(T/2) = vz(T/2) = 0
  3. Pseudo-arclength continuation along the halo family, and correction to
     a prescribed Jacobi constant (for energy-matched orbit pairs)

Every corrected orbit can be checked with ``periodicity_error``; the test
suite requires < 1e-9 after one full period.

History
-------
Versions up to v1.0.0 integrated to a fixed time T/2 and zeroed (vx, vz)
without enforcing the y = 0 crossing. Those "orbits" were not periodic
(return error ~0.3-0.4 non-dimensional) and small-amplitude cases collapsed
to z0 = 0. The Richardson guess also applied the in-plane frequency twice to
the velocities and dropped the Delta term of the amplitude constraint. All
three are fixed here.

References
----------
Richardson, D. L. (1980). Analytic construction of periodic orbits about the
  collinear points.  *Celestial Mechanics*, 22, 241–253.
Howell, K. C. (1984). Three-dimensional, periodic, 'halo' orbits.
  *Celestial Mechanics*, 32, 53–71.
Koon et al. (2008). *Dynamical Systems, the Three-Body Problem and Space
  Mission Design*. Springer.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import (
    MU_EARTH_MOON,
    cr3bp_eom,
    cr3bp_eom_stm,
    jacobi_constant,
    lagrange_points,
    propagate,
)

# ── Richardson 3rd-order initial guess ───────────────────────────────────────


def _gamma_L(lp: str, mu: float) -> float:
    """Distance from the libration point to the smaller primary.

    Computed from the 5th-degree Legendre polynomial equation used in
    Richardson's (1980) expansion.
    """
    from scipy.optimize import brentq

    lpts = lagrange_points(mu)
    x_moon = 1.0 - mu  # Moon position

    if lp == "L1":
        # γ₁ = 1 - μ - x_L1
        def eq(g): return g**5 - (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 + 2 * mu * g - mu
        return brentq(eq, 0.0, 1.0)
    elif lp == "L2":
        # γ₂ = x_L2 - (1 - μ)
        def eq(g): return g**5 + (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 - 2 * mu * g - mu
        return brentq(eq, 0.0, 1.0)
    else:
        raise ValueError("Richardson guess only for L1 or L2.")


def _richardson_coefficients(lp: str, mu: float) -> dict:
    """Coefficients of Richardson's third-order halo solution."""
    g = _gamma_L(lp, mu)
    if lp == "L1":
        def c(n): return (mu + (-1) ** n * (1 - mu) * g ** (n + 1) / (1 - g) ** (n + 1)) / g**3
    else:
        def c(n): return ((-1) ** n * mu + (-1) ** n * (1 - mu) * g ** (n + 1) / (1 + g) ** (n + 1)) / g**3
    c2, c3, c4 = c(2), c(3), c(4)

    # In-plane frequency: λ⁴ + (c2 − 2)λ² − (c2 − 1)(1 + 2c2) = 0
    lam = np.sqrt((-(c2 - 2) + np.sqrt((c2 - 2) ** 2 + 4 * (c2 - 1) * (1 + 2 * c2))) / 2)
    k = 2 * lam / (lam**2 + 1 - c2)
    Delta = lam**2 - c2

    d1 = 3 * lam**2 / k * (k * (6 * lam**2 - 1) - 2 * lam)
    d2 = 8 * lam**2 / k * (k * (11 * lam**2 - 1) - 2 * lam)
    a21 = 3 * c3 * (k**2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = -3 * c3 * lam / (4 * k * d1) * (3 * k**3 * lam - 6 * k * (k - lam) + 4)
    a24 = -3 * c3 * lam / (4 * k * d1) * (2 + 3 * k * lam)
    b21 = -3 * c3 * lam / (2 * d1) * (3 * k * lam - 4)
    b22 = 3 * c3 * lam / d1
    d21 = -c3 / (2 * lam**2)
    a31 = (-9 * lam / (4 * d2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2))
           + (9 * lam**2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k**2)))
    a32 = -1 / d2 * (9 * lam / 4 * (4 * c3 * (k * a24 - b22) + k * c4)
                     + 1.5 * (9 * lam**2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4))
    b31 = 3 / (8 * d2) * (8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k**2))
                          + (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2)))
    b32 = 1 / d2 * (9 * lam * (c3 * (k * b22 + d21 - 2 * a24) - c4)
                    + 3 / 8 * (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4))
    d31 = 3 / (64 * lam**2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lam**2) * (4 * c3 * (a23 - d21) + c4 * (4 + k**2))
    den = 2 * lam * (lam * (1 + k**2) - 2 * k)
    s1 = (1.5 * c3 * (2 * a21 * (k**2 - 2) - a23 * (k**2 + 2) - 2 * k * b21)
          - 3 / 8 * c4 * (3 * k**4 - 8 * k**2 + 8)) / den
    s2 = (1.5 * c3 * (2 * a22 * (k**2 - 2) + a24 * (k**2 + 2) + 2 * k * b22 + 5 * d21)
          + 3 / 8 * c4 * (12 - k**2)) / den
    l1 = -1.5 * c3 * (2 * a21 + a23 + 5 * d21) - 3 / 8 * c4 * (12 - k**2) + 2 * lam**2 * s1
    l2 = 1.5 * c3 * (a24 - 2 * a22) + 9 / 8 * c4 + 2 * lam**2 * s2
    return dict(gamma=g, lam=lam, k=k, Delta=Delta, a21=a21, a22=a22, a23=a23, a24=a24,
                a31=a31, a32=a32, b21=b21, b22=b22, b31=b31, b32=b32, d21=d21,
                d31=d31, d32=d32, s1=s1, s2=s2, l1=l1, l2=l2)


def richardson_guess(
    Az: float,
    lp: str = "L1",
    mu: float = MU_EARTH_MOON,
    northern: bool = True,
) -> tuple[np.ndarray, float]:
    """Third-order Richardson (1980) analytic initial guess for a halo orbit.

    Parameters
    ----------
    Az       : out-of-plane amplitude, non-dimensional (Earth–Moon distance
               units; Az = 0.05 is about 19 200 km)
    lp       : 'L1' or 'L2'
    mu       : mass parameter
    northern : northern (z0 > 0) or southern family member

    Returns
    -------
    state0 : (6,) state [x, 0, z, 0, vy, 0] at the xz-plane crossing
    T_half : half-period estimate π / (λ ω)

    Raises
    ------
    ValueError if Az is below the halo bifurcation amplitude (no real Ax).
    """
    if lp not in ("L1", "L2"):
        raise ValueError("Richardson guess only for L1 or L2.")
    c = _richardson_coefficients(lp, mu)
    g, lam, k = c["gamma"], c["lam"], c["k"]
    Az_g = Az / g                                   # Richardson length unit is γ
    Ax_sq = -(c["l2"] * Az_g**2 + c["Delta"]) / c["l1"]
    if Ax_sq <= 0:
        raise ValueError(f"No real in-plane amplitude for Az = {Az}.")
    Ax = np.sqrt(Ax_sq)
    omega = 1 + c["s1"] * Ax**2 + c["s2"] * Az_g**2
    dm = 1.0 if northern else -1.0

    # τ1 = 0: perpendicular xz-plane crossing
    x = (c["a21"] * Ax**2 + c["a22"] * Az_g**2 - Ax
         + (c["a23"] * Ax**2 - c["a24"] * Az_g**2)
         + (c["a31"] * Ax**3 - c["a32"] * Ax * Az_g**2))
    z = dm * (Az_g + c["d21"] * Ax * Az_g * (1 - 3) + (c["d32"] * Az_g * Ax**2 - c["d31"] * Az_g**3))
    vy = lam * omega * (k * Ax + 2 * (c["b21"] * Ax**2 - c["b22"] * Az_g**2)
                        + 3 * (c["b31"] * Ax**3 - c["b32"] * Ax * Az_g**2))

    x_L = lagrange_points(mu)[lp][0]
    state0 = np.array([x_L + g * x, 0.0, g * z, 0.0, g * vy, 0.0])
    return state0, np.pi / (lam * omega)


# ── Half-period map ───────────────────────────────────────────────────────────


def half_period_crossing(
    state0: np.ndarray,
    mu: float = MU_EARTH_MOON,
    t_skip: float = 0.3,
    t_max: float = 20.0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Integrate state + STM from the xz-plane to the next y = 0 crossing.

    Returns
    -------
    t_half : crossing time
    state  : (6,) state at the crossing
    stm    : (6, 6) Φ(t_half, 0)
    """
    y0 = np.concatenate([state0, np.eye(6).ravel()])
    first = solve_ivp(cr3bp_eom_stm, (0.0, t_skip), y0, args=(mu,),
                      method="DOP853", rtol=rtol, atol=atol)

    def y_zero(t, s, mu): return s[1]
    y_zero.terminal = True
    y_zero.direction = -np.sign(state0[4]) if state0[4] != 0 else 0.0

    sol = solve_ivp(cr3bp_eom_stm, (t_skip, t_max), first.y[:, -1], args=(mu,),
                    method="DOP853", rtol=rtol, atol=atol, events=y_zero)
    if sol.t_events[0].size == 0:
        raise RuntimeError("No return to the xz-plane within t_max.")
    yf = sol.y_events[0][0]
    return float(sol.t_events[0][0]), yf[:6].copy(), yf[6:].reshape(6, 6)


def _constraint_jacobian(state_h: np.ndarray, stm: np.ndarray, mu: float, free: list[int]) -> np.ndarray:
    """∂(vx, vz)|_{y=0} / ∂(free initial variables), including the change in crossing time."""
    f = cr3bp_eom(0.0, state_h, mu)
    return stm[[3, 5]][:, free] - np.outer([f[3], f[5]], stm[1, free]) / state_h[4]


def differential_correction(
    state0: np.ndarray,
    T_half: float | None = None,
    mu: float = MU_EARTH_MOON,
    tol: float = 1e-11,
    max_iter: int = 50,
    fix: str = "z0",
) -> tuple[np.ndarray, float]:
    """Single-shooting differential correction for a symmetric periodic orbit.

    The orbit starts perpendicular to the xz-plane, (y, vx, vz) = 0, and must
    return perpendicular to it: at the next y = 0 crossing vx = vz = 0.

    Parameters
    ----------
    state0   : (6,) initial guess [x0, 0, z0, 0, vy0, 0]
    T_half   : unused, kept for backward compatibility (the crossing time is
               found by event detection)
    mu       : mass parameter
    tol      : tolerance on max(|vx|, |vz|) at the crossing
    max_iter : maximum Newton iterations
    fix      : 'z0' (correct x0, vy0; good for small/medium halos) or
               'x0' (correct z0, vy0; good near the NRHO end of the family)

    Returns
    -------
    state_corrected : (6,) corrected initial state
    T_half          : half-period

    Raises
    ------
    RuntimeError if the iteration does not converge.
    """
    state = np.array(state0, dtype=float).copy()
    state[[1, 3, 5]] = 0.0
    free = [0, 4] if fix == "z0" else [2, 4]
    for _ in range(max_iter):
        t_h, s_h, stm = half_period_crossing(state, mu)
        F = np.array([s_h[3], s_h[5]])
        if np.max(np.abs(F)) < tol:
            return state, t_h
        DF = _constraint_jacobian(s_h, stm, mu, free)
        dx = np.linalg.solve(DF, -F)
        step = np.linalg.norm(dx)
        if step > 0.02:                              # damp large Newton steps
            dx *= 0.02 / step
        state[free] += dx
    raise RuntimeError(
        f"Differential correction did not converge in {max_iter} iterations "
        f"(residual {np.max(np.abs(F)):.2e})."
    )


def periodicity_error(state0: np.ndarray, T: float, mu: float = MU_EARTH_MOON) -> float:
    """‖x(T) − x(0)‖ after one full period (non-dimensional)."""
    sol = propagate(state0, (0.0, T), mu=mu)
    return float(np.linalg.norm(sol.y[:, -1] - state0))


def stability_index(state0: np.ndarray, T: float, mu: float = MU_EARTH_MOON) -> float:
    """ν = (|λ_max| + 1/|λ_max|) / 2 from the monodromy matrix (ν > 1: unstable)."""
    from .dynamics import propagate_stm
    _, M = propagate_stm(state0, (0.0, T), mu)
    lmax = np.max(np.abs(np.linalg.eigvals(M)))
    return float(0.5 * (lmax + 1.0 / lmax))


def z_amplitude(state0: np.ndarray, T: float, mu: float = MU_EARTH_MOON) -> float:
    """Maximum |z| over one period (the physical out-of-plane amplitude)."""
    sol = propagate(state0, (0.0, T), mu=mu, t_eval=np.linspace(0.0, T, 400))
    return float(np.max(np.abs(sol.y[2])))


# ── Pseudo-arclength continuation ─────────────────────────────────────────────

_FREE = [0, 2, 4]  # x0, z0, vy0


def _family_tangent(state: np.ndarray, mu: float, prev: np.ndarray | None) -> np.ndarray:
    _, s_h, stm = half_period_crossing(state, mu)
    DF = _constraint_jacobian(s_h, stm, mu, _FREE)
    tau = np.linalg.svd(DF)[2][-1]
    if prev is not None and np.dot(tau, prev) < 0:
        tau = -tau
    return tau


def continuation_step(
    state0: np.ndarray,
    T_half: float,
    ds: float,
    tangent: np.ndarray,
    mu: float = MU_EARTH_MOON,
    tol: float = 1e-11,
    max_iter: int = 30,
) -> tuple[np.ndarray, float, np.ndarray]:
    """One pseudo-arclength step in X = (x0, z0, vy0).

    Predictor X* = X + ds·τ; corrector solves
        vx(T/2) = 0,  vz(T/2) = 0,  (X − X_prev)·τ − ds = 0.

    Returns
    -------
    new_state0, new_T_half, new_tangent
    """
    X_prev = state0[_FREE].copy()
    state = state0.copy()
    state[_FREE] = X_prev + ds * tangent
    for _ in range(max_iter):
        t_h, s_h, stm = half_period_crossing(state, mu)
        X = state[_FREE]
        G = np.array([s_h[3], s_h[5], np.dot(X - X_prev, tangent) - ds])
        if np.max(np.abs(G)) < tol:
            new_tangent = _family_tangent(state, mu, tangent)
            return state, t_h, new_tangent
        DG = np.vstack([_constraint_jacobian(s_h, stm, mu, _FREE), tangent])
        state[_FREE] = X + np.linalg.solve(DG, -G)
    raise RuntimeError("Continuation corrector did not converge.")


def _family_member(state0: np.ndarray, T_half: float, mu: float, Az_nominal: float | None = None) -> dict:
    T = 2.0 * T_half
    return {
        "state0": state0.copy(),
        "T_half": T_half,
        "T": T,
        "Az": z_amplitude(state0, T, mu),
        "C": jacobi_constant(state0, mu),
        "nu": stability_index(state0, T, mu),
    }


def compute_halo_family(
    Az_start: float,
    Az_end: float,
    n_orbits: int = 20,
    lp: str = "L1",
    mu: float = MU_EARTH_MOON,
    northern: bool = True,
    ds: float = 0.005,
    max_steps: int = 400,
) -> list[dict]:
    """Trace a halo family by pseudo-arclength continuation.

    Starts from a corrected Richardson orbit at ``Az_start`` and continues
    in the direction of increasing out-of-plane amplitude until the
    initial out-of-plane coordinate |z0| reaches ``Az_end`` (or ``max_steps`` steps). Returns about
    ``n_orbits`` members spaced evenly along the arclength.

    Each member is a dict with keys
        'state0', 'T_half', 'T', 'Az' (max |z| over the orbit),
        'C' (Jacobi constant), 'nu' (stability index).
    """
    state, t_h = differential_correction(
        richardson_guess(Az_start, lp=lp, mu=mu, northern=northern)[0], mu=mu
    )
    tangent = _family_tangent(state, mu, None)
    if np.sign(tangent[1]) != np.sign(state[2]):     # grow |z0|
        tangent = -tangent

    path = [(state.copy(), t_h)]
    for _ in range(max_steps):
        state, t_h, tangent = continuation_step(state, t_h, ds, tangent, mu=mu)
        path.append((state.copy(), t_h))
        if abs(state[2]) >= Az_end:
            break

    idx = np.unique(np.linspace(0, len(path) - 1, min(n_orbits, len(path))).round().astype(int))
    return [_family_member(path[i][0], path[i][1], mu) for i in idx]


def correct_to_jacobi(
    state0: np.ndarray,
    C_target: float,
    mu: float = MU_EARTH_MOON,
    tol: float = 1e-11,
    max_iter: int = 30,
) -> tuple[np.ndarray, float]:
    """Correct a family member so its Jacobi constant equals ``C_target``.

    Solves vx(T/2) = vz(T/2) = 0 and C(X) = C_target for X = (x0, z0, vy0).
    Used to build energy-matched L1/L2 orbit pairs.
    """
    from .dynamics import distances
    state = state0.copy()
    for _ in range(max_iter):
        t_h, s_h, stm = half_period_crossing(state, mu)
        C = jacobi_constant(state, mu)
        G = np.array([s_h[3], s_h[5], C - C_target])
        if np.max(np.abs(G)) < tol:
            return state, t_h
        x, _, z, _, vy, _ = state
        r1, r2 = distances(state, mu)
        Ox = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        Oz = -(1 - mu) * z / r1**3 - mu * z / r2**3
        dC = np.array([2 * Ox, 2 * Oz, -2 * vy])
        DG = np.vstack([_constraint_jacobian(s_h, stm, mu, _FREE), dC])
        dx = np.linalg.solve(DG, -G)
        if np.linalg.norm(dx) > 0.01:
            dx *= 0.01 / np.linalg.norm(dx)
        state[_FREE] += dx
    raise RuntimeError("Jacobi-constant correction did not converge.")


def orbit_at_jacobi(
    family: list[dict],
    C_target: float,
    mu: float = MU_EARTH_MOON,
    dC_max: float = 0.002,
) -> tuple[np.ndarray, float]:
    """Halo with Jacobi constant ``C_target``.

    Only the first monotonic stretch of the family (in C) is used. Starts
    from the family member nearest in C and walks to ``C_target`` in
    steps of at most ``dC_max``, correcting onto each intermediate C
    (natural-parameter continuation in C), so the solution stays on the
    same branch.
    """
    Cs = np.array([m["C"] for m in family])
    # use the first monotonic stretch of the family, so that a C value that
    # occurs twice (the family turns back in C) is resolved on the first branch
    dC = np.sign(np.diff(Cs))
    turn = np.argmax(dC != dC[0]) if np.any(dC != dC[0]) else len(Cs) - 1
    family, Cs = family[: turn + 1], Cs[: turn + 1]
    if not (Cs.min() <= C_target <= Cs.max()):
        raise ValueError(f"C = {C_target} outside family range [{Cs.min():.5f}, {Cs.max():.5f}].")
    m = family[int(np.argmin(np.abs(Cs - C_target)))]
    state, t_h = m["state0"].copy(), m["T_half"]
    n = max(1, int(np.ceil(abs(C_target - m["C"]) / dC_max)))
    for C in np.linspace(m["C"], C_target, n + 1)[1:]:
        state, t_h = correct_to_jacobi(state, C, mu)
    return state, t_h
