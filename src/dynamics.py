"""
dynamics.py
===========
Equations of motion, state transition matrix (STM) propagation,
and energy integrals for the Circular Restricted Three-Body Problem (CR3BP).

All quantities are non-dimensionalized:
  - Length  : L*  = Earth–Moon distance  ≈ 384 400 km
  - Time    : T*  = 1 / n,  n = mean motion of the Moon  ≈ 2.661 × 10⁻⁶ rad/s
  - Mass    : M*  = m_Earth + m_Moon

Earth–Moon mass parameter:
  μ = m_Moon / (m_Earth + m_Moon)  ≈ 0.012150585609

References
----------
Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D. (2008).
  *Dynamical Systems, the Three-Body Problem and Space Mission Design*.
  Springer.
Szebehely, V. (1967). *Theory of Orbits*. Academic Press.
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── Physical constants ────────────────────────────────────────────────────────

MU_EARTH_MOON: float = 0.012150585609  # Earth–Moon mass parameter
L_STAR: float = 384_400.0              # km, Earth–Moon characteristic length
T_STAR: float = 375_699.8              # s,  1/n (non-dim time unit)
V_STAR: float = L_STAR / T_STAR        # km/s, characteristic velocity

# ── Core CR3BP functions ──────────────────────────────────────────────────────


def distances(state: np.ndarray, mu: float) -> tuple[float, float]:
    """Distances from the primary bodies to the spacecraft.

    Parameters
    ----------
    state : (6,) array  [x, y, z, vx, vy, vz] in rotating frame
    mu    : mass parameter of the smaller body

    Returns
    -------
    r1, r2 : distances from larger body (1−μ on left) and smaller body (μ on right)
    """
    x, y, z = state[0], state[1], state[2]
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
    return r1, r2


def pseudo_potential(state: np.ndarray, mu: float) -> float:
    """Augmented (pseudo-) potential Ω for the CR3BP.

    Ω = ½(x² + y²) + (1−μ)/r₁ + μ/r₂ + ½μ(1−μ)
    The constant last term is often omitted; we include it for completeness.
    """
    x, y = state[0], state[1]
    r1, r2 = distances(state, mu)
    return 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2 + 0.5 * mu * (1.0 - mu)


def jacobi_constant(state: np.ndarray, mu: float) -> float:
    """Jacobi constant C = 2Ω − v².

    This is the only integral of motion in the CR3BP; C is conserved along
    any trajectory.

    Parameters
    ----------
    state : (6,) array  [x, y, z, vx, vy, vz]
    mu    : mass parameter

    Returns
    -------
    C : Jacobi constant (dimensionless)
    """
    vx, vy, vz = state[3], state[4], state[5]
    v2 = vx**2 + vy**2 + vz**2
    omega = pseudo_potential(state, mu)
    return 2.0 * omega - v2


def cr3bp_eom(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations of motion (rotating frame, non-dimensional).

    ẍ − 2ẏ = ∂Ω/∂x
    ÿ + 2ẋ = ∂Ω/∂y
    z̈      = ∂Ω/∂z

    Parameters
    ----------
    t     : time (unused; system is autonomous)
    state : (6,) array [x, y, z, vx, vy, vz]
    mu    : mass parameter

    Returns
    -------
    dstate/dt : (6,) array [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    r1, r2 = distances(state, mu)
    r1_3 = r1**3
    r2_3 = r2**3

    c1 = (1.0 - mu) / r1_3
    c2 = mu / r2_3

    ax = 2.0 * vy + x - c1 * (x + mu) - c2 * (x - 1.0 + mu)
    ay = -2.0 * vx + y - c1 * y - c2 * y
    az = -c1 * z - c2 * z

    return np.array([vx, vy, vz, ax, ay, az])


# ── State Transition Matrix (STM) ─────────────────────────────────────────────


def cr3bp_eom_stm(t: float, state_stm: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations of motion augmented with the STM.

    The variational equations are

        dΦ/dt = A(t) Φ

    where A(t) is the 6×6 Jacobian of the EOM evaluated at the current state.
    The full state vector has 6 + 36 = 42 components.

    Parameters
    ----------
    t         : time
    state_stm : (42,) array — first 6 are the state, next 36 are Φ (row-major)
    mu        : mass parameter

    Returns
    -------
    d(state_stm)/dt : (42,) array
    """
    state = state_stm[:6]
    phi = state_stm[6:].reshape(6, 6)

    x, y, z = state[0], state[1], state[2]
    r1, r2 = distances(state, mu)
    r1_3 = r1**3
    r2_3 = r2**3
    r1_5 = r1**5
    r2_5 = r2**5

    # Second-order partial derivatives of Ω
    uxx = (
        1.0
        - (1.0 - mu) / r1_3
        - mu / r2_3
        + 3.0 * (1.0 - mu) * (x + mu) ** 2 / r1_5
        + 3.0 * mu * (x - 1.0 + mu) ** 2 / r2_5
    )
    uyy = (
        1.0
        - (1.0 - mu) / r1_3
        - mu / r2_3
        + 3.0 * (1.0 - mu) * y**2 / r1_5
        + 3.0 * mu * y**2 / r2_5
    )
    uzz = (
        -(1.0 - mu) / r1_3
        - mu / r2_3
        + 3.0 * (1.0 - mu) * z**2 / r1_5
        + 3.0 * mu * z**2 / r2_5
    )
    uxy = (
        3.0 * (1.0 - mu) * (x + mu) * y / r1_5
        + 3.0 * mu * (x - 1.0 + mu) * y / r2_5
    )
    uxz = (
        3.0 * (1.0 - mu) * (x + mu) * z / r1_5
        + 3.0 * mu * (x - 1.0 + mu) * z / r2_5
    )
    uyz = (
        3.0 * (1.0 - mu) * y * z / r1_5
        + 3.0 * mu * y * z / r2_5
    )

    # Jacobian A of the EOM
    A = np.array([
        [0,   0,   0,   1,  0,  0],
        [0,   0,   0,   0,  1,  0],
        [0,   0,   0,   0,  0,  1],
        [uxx, uxy, uxz, 0,  2,  0],
        [uxy, uyy, uyz, -2, 0,  0],
        [uxz, uyz, uzz, 0,  0,  0],
    ])

    dphi_dt = A @ phi
    dstate = cr3bp_eom(t, state, mu)
    return np.concatenate([dstate, dphi_dt.ravel()])


# ── Integration helpers ────────────────────────────────────────────────────────


def propagate(
    state0: np.ndarray,
    t_span: tuple[float, float],
    mu: float = MU_EARTH_MOON,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    t_eval: np.ndarray | None = None,
    events=None,
) -> object:
    """Integrate the CR3BP equations of motion.

    Parameters
    ----------
    state0  : (6,) initial state
    t_span  : (t0, tf) integration interval
    mu      : mass parameter  (default: Earth–Moon)
    rtol    : relative tolerance for the integrator
    atol    : absolute tolerance for the integrator
    t_eval  : optional array of output times
    events  : optional event functions for solve_ivp

    Returns
    -------
    OdeResult from scipy.integrate.solve_ivp
    """
    return solve_ivp(
        cr3bp_eom,
        t_span,
        state0,
        args=(mu,),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        dense_output=True,
        events=events,
    )


def propagate_stm(
    state0: np.ndarray,
    t_span: tuple[float, float],
    mu: float = MU_EARTH_MOON,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    t_eval: np.ndarray | None = None,
) -> tuple[object, np.ndarray]:
    """Integrate the CR3BP EOM together with the STM.

    Parameters
    ----------
    state0  : (6,) initial state
    t_span  : (t0, tf) integration interval
    mu      : mass parameter

    Returns
    -------
    sol : OdeResult (first 6 rows are the state trajectory)
    stm : (6, 6) state transition matrix evaluated at t_span[1]
    """
    phi0 = np.eye(6)
    state_stm0 = np.concatenate([state0, phi0.ravel()])
    sol = solve_ivp(
        cr3bp_eom_stm,
        t_span,
        state_stm0,
        args=(mu,),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        dense_output=True,
    )
    stm = sol.y[6:, -1].reshape(6, 6)
    return sol, stm


# ── Lagrange point utilities ───────────────────────────────────────────────────


def lagrange_points(mu: float = MU_EARTH_MOON) -> dict[str, np.ndarray]:
    """Compute the five Lagrange (libration) points.

    L1, L2, L3 are on the x-axis; L4 and L5 form equilateral triangles.

    Parameters
    ----------
    mu : mass parameter

    Returns
    -------
    dict mapping 'L1' … 'L5' to (6,) state vectors (velocities = 0)
    """
    from scipy.optimize import brentq

    def collinear_eq(x: float, sign: int) -> float:
        """Equation whose root gives a collinear Lagrange point x-coordinate."""
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        return (
            x
            - (1.0 - mu) * np.sign(x + mu) / r1**2
            - mu * np.sign(x - 1.0 + mu) / r2**2
        )

    # L1 — between the two primaries
    x_L1 = brentq(collinear_eq, -mu + 1e-6, 1.0 - mu - 1e-6, args=(0,))
    # L2 — beyond the smaller primary (Moon side)
    x_L2 = brentq(collinear_eq, 1.0 - mu + 1e-6, 2.0, args=(0,))
    # L3 — beyond the larger primary (Earth side, negative x)
    x_L3 = brentq(collinear_eq, -2.0, -mu - 1e-6, args=(0,))

    def to_state(x, y=0.0):
        return np.array([x, y, 0.0, 0.0, 0.0, 0.0])

    return {
        "L1": to_state(x_L1),
        "L2": to_state(x_L2),
        "L3": to_state(x_L3),
        "L4": to_state(0.5 - mu,  np.sqrt(3.0) / 2.0),
        "L5": to_state(0.5 - mu, -np.sqrt(3.0) / 2.0),
    }
