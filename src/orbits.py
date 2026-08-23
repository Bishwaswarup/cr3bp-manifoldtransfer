"""
orbits.py
=========
Halo orbit computation via:
  1. Third-order Richardson analytic initial guess
  2. Differential correction (single-shooting) to obtain a numerically exact
     periodic orbit
  3. Pseudo-arclength continuation to trace the halo/Lyapunov family

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
from scipy.optimize import fsolve

from .dynamics import (
    MU_EARTH_MOON,
    lagrange_points,
    propagate,
    propagate_stm,
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


def richardson_guess(
    Az: float,
    lp: str = "L1",
    mu: float = MU_EARTH_MOON,
    northern: bool = True,
) -> np.ndarray:
    """Third-order Richardson (1980) analytic initial guess for a halo orbit.

    Parameters
    ----------
    Az      : out-of-plane amplitude (non-dimensional)
    lp      : 'L1' or 'L2'
    mu      : mass parameter
    northern: if True, northern halo (z > 0 at t=0); else southern

    Returns
    -------
    state0 : (6,) initial state [x, y=0, z, vx=0, vy, vz=0] at the
             xz-plane crossing (y=0, vy≠0 by convention)
    T_half : half-period estimate
    """
    gamma = _gamma_L(lp, mu)
    lpts = lagrange_points(mu)
    x_L = lpts[lp][0]

    # Collinear libration point mass ratios (Richardson notation)
    c = {}
    sign = 1 if lp == "L1" else -1
    for n in range(2, 5):
        c[n] = (
            (1.0 / gamma**3)
            * (
                mu + sign * (-1) ** n * (1.0 - mu) * gamma ** (n + 1) / (1.0 - gamma) ** (n + 1)
            )
        )

    c2, c3, c4 = c[2], c[3], c[4]

    # Frequency λ (in-plane)
    lam_sq_roots = np.roots([1, 0, -(c2 - 2), 0, -(1 + 2 * c2)])
    lam_sq_pos = [r for r in lam_sq_roots if np.isreal(r) and r.real > 0]
    lam = float(np.sqrt(np.min([r.real for r in lam_sq_pos])))

    k = 2 * lam / (lam**2 + 1 - c2)

    # 3rd-order amplitudes
    d1 = 3 * lam**2 / k * (k * (6 * lam**2 - 1) - 2 * lam)
    d2 = 8 * lam**2 / k * (k * (11 * lam**2 - 1) - 2 * lam)

    a21 = 3 * c3 * (k**2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = -3 * c3 * lam / (4 * k * d1) * (3 * k**3 * lam - 6 * k * (1 - c2) + 4)
    a24 = -3 * c3 * lam / (4 * k * d1) * (2 + 3 * k * lam)

    b21 = -3 * c3 * lam / (2 * d1) * (3 * k * lam - 4)
    b22 = 3 * c3 * lam / d1

    d21 = -c3 / (2 * lam**2)

    a31 = (-9 * lam / (4 * d2)) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2))
    a32 = (1 / (2 * d2)) * (9 * lam * (4 * c3 * (k * a24 - b22) + k * c4) + 3 * c3 * (2 * a23 - k * b21))

    b31 = (3 / (8 * d2)) * (8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4) + (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2)))
    b32 = (1 / d2) * (9 * lam * (c3 * (k * b22 + d21 - 2 * a24) - c4) + (9 * lam**2 + 1 + 2 * c2) / 8 * (4 * c3 * (k * a24 - b22) + k * c4))

    d31 = 3 / (64 * lam**2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lam**2) * (4 * c3 * (a23 - d21) + c4 * (4 + k**2))

    s1 = 1 / (2 * lam * (lam * (1 + k**2) - 2 * k)) * (
        3 * c3 / 2 * (2 * a21 * (k**2 - 2) - a23 * (k**2 + 2) - 2 * k * b21)
        - 3 * c4 / 8 * (3 * k**4 - 8 * k**2 + 8)
    )
    s2 = 1 / (2 * lam * (lam * (1 + k**2) - 2 * k)) * (
        3 * c3 / 2 * (2 * a22 * (k**2 - 2) + a24 * (k**2 + 2) + 2 * k * b22 + 5 * d21)
        + 3 * c4 / 8 * (12 - k**2)
    )

    # In-plane amplitude Ax derived from Az via the frequency constraint
    # (avoids secular terms in 3rd order expansion)
    l1 = a21 + 2 * lam**2 * s1 - (b21 + 2 * k * s1)
    l2 = a22 + 2 * lam**2 * s2 + (b22 + 2 * k * s2)
    # Polynomial for Ax from Richardson eq. (26)
    # Δ Ax² = l1 Az² => for simplicity use Ax from direct 2nd order:
    Ax = np.sqrt(np.abs(-l2 / l1) * Az**2) if np.abs(l1) > 1e-12 else Az

    # Phase φ = 0  (x-z crossing)
    tau = 0.0
    m = 1 if northern else 3  # northern: m=1, delta_n > 0

    # Position
    delta = 2 - m  # ±1 for northern/southern
    x0 = (
        a21 * Ax**2
        + a22 * Az**2
        - Ax * np.cos(tau)
        + (a23 * Ax**2 - a24 * Az**2) * np.cos(2 * tau)
        + (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3 * tau)
    )
    y0 = (
        k * Ax * np.sin(tau)
        + (b21 * Ax**2 - b22 * Az**2) * np.sin(2 * tau)
        + (b31 * Ax**3 - b32 * Ax * Az**2) * np.sin(3 * tau)
    )
    z0 = (
        delta * Az * np.cos(tau)
        + delta * d21 * Ax * Az * (np.cos(2 * tau) - 3)
        + delta * (d32 * Az * Ax**2 - d31 * Az**3) * np.cos(3 * tau)
    )
    # Velocity
    dx0 = (
        Ax * lam * np.sin(tau)
        - 2 * (a23 * Ax**2 - a24 * Az**2) * lam * np.sin(2 * tau)
        - 3 * (a31 * Ax**3 - a32 * Ax * Az**2) * lam * np.sin(3 * tau)
    )
    dy0 = (
        k * Ax * lam * np.cos(tau)
        + 2 * (b21 * Ax**2 - b22 * Az**2) * lam * np.cos(2 * tau)
        + 3 * (b31 * Ax**3 - b32 * Ax * Az**2) * lam * np.cos(3 * tau)
    )
    dz0 = (
        -delta * Az * lam * np.sin(tau)
        - 2 * delta * d21 * Ax * Az * lam * np.sin(2 * tau)
        - 3 * delta * (d32 * Az * Ax**2 - d31 * Az**3) * lam * np.sin(3 * tau)
    )

    # Transform from libration-point-centered to system barycentric coords
    sign_gamma = -1 if lp == "L1" else 1
    x_sys = x_L + gamma * x0
    y_sys = gamma * y0
    z_sys = gamma * z0
    vx_sys = gamma * dx0 * lam  # lam absorbed; correct sign
    vy_sys = gamma * dy0 * lam
    vz_sys = gamma * dz0 * lam

    # Note: Richardson gives dimensionless velocity in lam*gamma units;
    # we use the convention dτ/dt = 1, so velocities are already in
    # non-dimensional units per non-dimensional time.
    # Reset vx to 0 for the xz crossing convention used in diff-correction.
    state0 = np.array([x_sys, y_sys, z_sys, 0.0, vy_sys, 0.0])

    T_half_est = np.pi / lam  # half period estimate

    return state0, T_half_est


# ── Differential correction (single-shooting) ─────────────────────────────────


def differential_correction(
    state0: np.ndarray,
    T_half: float,
    mu: float = MU_EARTH_MOON,
    tol: float = 1e-10,
    max_iter: int = 50,
    fix_xz: bool = True,
) -> tuple[np.ndarray, float]:
    """Single-shooting differential correction to find a periodic halo orbit.

    Convention (xz-plane crossing):
      - state0 has y=0 and vx=0
      - at t = T_half the orbit should return to y=0 with vx=0

    Free variables: z₀, ẏ₀, T/2
    Constraints:    Δvₓ = 0, Δvz = 0  at y = 0 crossing

    Parameters
    ----------
    state0   : (6,) initial state with y=0, vx=0
    T_half   : initial half-period estimate
    mu       : mass parameter
    tol      : convergence tolerance on the correction norm
    max_iter : maximum Newton iterations
    fix_xz   : if True, use [z0, vy0, T/2] as free variables (default)

    Returns
    -------
    state_corrected : (6,) corrected initial state
    T_half_corrected : corrected half-period
    """
    state = state0.copy()
    T = T_half

    for iteration in range(max_iter):
        # Define y=0 crossing event
        def y_zero(t, s, mu): return s[1]
        y_zero.terminal = True
        y_zero.direction = -1.0  # look for negative crossing (going from +y to -y)

        sol, stm = propagate_stm(state, (0.0, 2.0 * T), mu)

        # Find first y=0 crossing after t=0
        # Use the half-period estimate: integrate to T, then refine with event
        sol_half, stm_half = propagate_stm(state, (0.0, T), mu)
        state_half = sol_half.y[:6, -1]
        stm_T = stm_half

        # Residuals: vx and vz at the xz-plane crossing should be 0
        vx_f = state_half[3]
        vz_f = state_half[5]

        err = np.array([vx_f, vz_f])
        if np.linalg.norm(err) < tol:
            break

        # EOM at final state (for dT derivatives)
        f_f = np.zeros(6)
        f_f[:6] = _cr3bp_eom_simple(state_half, mu)

        # Jacobian: ∂(vx_f, vz_f) / ∂(z0, vy0, T)
        # From STM: ∂vx_f/∂z0 = stm[3,2], ∂vx_f/∂vy0 = stm[3,4]
        # ∂vx_f/∂T = f_f[3]  (EOM at final point)
        J = np.array([
            [stm_T[3, 2], stm_T[3, 4], f_f[3]],
            [stm_T[5, 2], stm_T[5, 4], f_f[5]],
        ])

        # Minimum-norm correction (underdetermined system)
        delta = -J.T @ np.linalg.lstsq(J @ J.T, err, rcond=None)[0]

        state[2] += delta[0]  # z0
        state[4] += delta[1]  # vy0
        T += delta[2]         # T/2
    else:
        import warnings
        warnings.warn(
            f"Differential correction did not converge in {max_iter} iterations. "
            f"Residual = {np.linalg.norm(err):.2e}"
        )

    return state, T


def _cr3bp_eom_simple(state: np.ndarray, mu: float) -> np.ndarray:
    """Thin wrapper for the CR3BP EOM without time argument."""
    from .dynamics import cr3bp_eom
    return cr3bp_eom(0.0, state, mu)


# ── Pseudo-arclength continuation ─────────────────────────────────────────────


def continuation_step(
    state0: np.ndarray,
    T_half: float,
    ds: float,
    tangent: np.ndarray,
    mu: float = MU_EARTH_MOON,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float, np.ndarray]:
    """One pseudo-arclength continuation step.

    Takes a predictor step along the tangent direction in (state0, T_half)
    space, then corrects back onto the solution manifold while holding the
    arclength constraint fixed.

    Parameters
    ----------
    state0  : (6,) converged initial state from the previous solution
    T_half  : converged half-period from the previous solution
    ds      : arclength step size
    tangent : (3,) unit tangent vector in (z0, vy0, T) subspace
    mu      : mass parameter
    tol     : convergence tolerance for the corrector

    Returns
    -------
    new_state0  : (6,) corrected initial state
    new_T_half  : corrected half-period
    new_tangent : (3,) updated tangent (for next step)
    """
    # Free variables: p = [z0, vy0, T_half]
    p_prev = np.array([state0[2], state0[4], T_half])
    p_pred = p_prev + ds * tangent

    # Build a full state from the predictor
    s_pred = state0.copy()
    s_pred[2] = p_pred[0]
    s_pred[4] = p_pred[1]
    T_pred = p_pred[2]

    # Corrector: augment residual with the arclength constraint
    def residual(p_free):
        s = state0.copy()
        s[2], s[4] = p_free[0], p_free[1]
        T = p_free[2]
        sol_half, stm_T = propagate_stm(s, (0.0, T), mu)
        state_half = sol_half.y[:6, -1]
        eq1 = state_half[3]                         # vx_f = 0
        eq2 = state_half[5]                         # vz_f = 0
        eq3 = np.dot(p_free - p_prev, tangent) - ds  # arclength
        return np.array([eq1, eq2, eq3])

    from scipy.optimize import fsolve as _fsolve
    p_corr, info, ier, msg = _fsolve(residual, p_pred, full_output=True)
    if ier != 1:
        import warnings
        warnings.warn(f"Continuation corrector did not converge: {msg}")

    s_new = state0.copy()
    s_new[2], s_new[4] = p_corr[0], p_corr[1]
    T_new = p_corr[2]

    # Update tangent: recompute Jacobian at the new point and get null vector
    sol_half, stm_T = propagate_stm(s_new, (0.0, T_new), mu)
    state_half = sol_half.y[:6, -1]
    f_f = _cr3bp_eom_simple(state_half, mu)

    J = np.array([
        [stm_T[3, 2], stm_T[3, 4], f_f[3]],
        [stm_T[5, 2], stm_T[5, 4], f_f[5]],
    ])
    # Null vector of J (pseudo-arclength tangent)
    _, _, Vt = np.linalg.svd(J)
    new_tangent = Vt[-1, :]  # last right singular vector
    # Ensure consistent direction
    if np.dot(new_tangent, tangent) < 0:
        new_tangent = -new_tangent

    return s_new, T_new, new_tangent


def compute_halo_family(
    Az_start: float,
    Az_end: float,
    n_orbits: int = 20,
    lp: str = "L1",
    mu: float = MU_EARTH_MOON,
    northern: bool = True,
    ds: float = 0.005,
) -> list[dict]:
    """Generate a family of halo orbits by continuation.

    Starts from a Richardson initial guess at Az_start and steps toward
    Az_end using pseudo-arclength continuation.

    Parameters
    ----------
    Az_start : starting out-of-plane amplitude (non-dim)
    Az_end   : ending out-of-plane amplitude (non-dim)
    n_orbits : number of orbits to compute
    lp       : 'L1' or 'L2'
    mu       : mass parameter
    northern : northern (z>0) family
    ds       : arclength step size

    Returns
    -------
    list of dicts, each with keys:
        'state0'  : (6,) initial state
        'T_half'  : half period
        'T'       : full period
        'Az'      : estimated out-of-plane amplitude
        'C'       : Jacobi constant
    """
    from .dynamics import jacobi_constant

    results = []

    # First orbit from Richardson guess
    state0, T_half = richardson_guess(Az_start, lp=lp, mu=mu, northern=northern)
    state0, T_half = differential_correction(state0, T_half, mu=mu)
    results.append({
        "state0": state0.copy(),
        "T_half": T_half,
        "T": 2.0 * T_half,
        "Az": Az_start,
        "C": jacobi_constant(state0, mu),
    })

    # Initial tangent from Richardson
    _, T_half_next = richardson_guess(Az_start + 0.001, lp=lp, mu=mu, northern=northern)
    tangent0 = np.array([
        0.0,                    # dz0 direction
        0.0,                    # dvy0 direction (will be updated)
        T_half_next - T_half,   # dT direction
    ])
    tangent0 /= (np.linalg.norm(tangent0) + 1e-30)

    # Recompute proper tangent from the Jacobian null vector
    sol_half, stm_T = propagate_stm(state0, (0.0, T_half), mu)
    state_half = sol_half.y[:6, -1]
    f_f = _cr3bp_eom_simple(state_half, mu)
    J = np.array([
        [stm_T[3, 2], stm_T[3, 4], f_f[3]],
        [stm_T[5, 2], stm_T[5, 4], f_f[5]],
    ])
    _, _, Vt = np.linalg.svd(J)
    tangent = Vt[-1, :]
    # Orient toward increasing Az (z0 grows)
    if tangent[0] < 0:
        tangent = -tangent

    prev_state = state0.copy()
    prev_T = T_half
    prev_tangent = tangent

    for i in range(1, n_orbits):
        try:
            new_state, new_T, new_tangent = continuation_step(
                prev_state, prev_T, ds, prev_tangent, mu=mu
            )
            Az_est = abs(new_state[2])  # z0 at crossing ≈ Az
            results.append({
                "state0": new_state.copy(),
                "T_half": new_T,
                "T": 2.0 * new_T,
                "Az": Az_est,
                "C": jacobi_constant(new_state, mu),
            })
            prev_state = new_state
            prev_T = new_T
            prev_tangent = new_tangent
        except Exception as e:
            import warnings
            warnings.warn(f"Continuation step {i} failed: {e}")
            break

    return results
