"""
transfer.py
===========
Transfer design tools for the CR3BP manifold-assisted low-energy transfer.

Status
------
The repository's results use ``connections.py`` (energy-matched L1/L2
patches) and ``reentry.py``. The two-body TLI estimate in
``leo_to_manifold_dv`` / ``full_transfer_budget`` is a rough Hohmann-style
approximation (it circularises at the insertion radius instead of matching
the manifold velocity) and the low-thrust shooting leg is experimental and
untested; neither is used for any figure or number in the README.

Implements:
  1. Full ΔV budget computation for a ballistic manifold transfer (two impulses:
     departure from parking orbit + arrival Poincaré ΔV patch).
  2. Optional low-thrust leg: continuous tangential thrust connecting a LEO/GTO
     parking orbit to the manifold insertion point via a shooting method.

ΔV conventions
--------------
All ΔV values are in non-dimensional units unless the *_km_s variant is given.
Conversion: 1 nd velocity = V_STAR km/s  (≈ 1.0232 km/s for Earth–Moon).

References
----------
Conley, C. C. (1968). Low energy transit orbits in the restricted three-body
  problem. *SIAM Journal on Applied Mathematics*, 16(4), 732–746.
Lo, M. W., & Ross, S. D. (1998). *The Lunar L1 Gateway: Portal to the Stars
  and Beyond*. AIAA Space 2001 Conference.
Parker, J. S., & Anderson, R. L. (2014).
  *Low-Energy Lunar Trajectory Design*. JPL.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from .dynamics import (
    MU_EARTH_MOON,
    V_STAR,
    L_STAR,
    propagate,
    cr3bp_eom,
    jacobi_constant,
)


# ── Parking orbit helpers ──────────────────────────────────────────────────────


def circular_orbit_velocity(r: float, mu: float = MU_EARTH_MOON) -> float:
    """Non-dimensional circular orbit speed at radius r from Earth (primary 1).

    In the CR3BP rotating frame, a circular parking orbit at r is not truly
    circular (the Coriolis and centrifugal terms perturb it), but for the
    purposes of computing a departure ΔV we use the two-body approximation:

        v_circ = sqrt((1−μ) / r)

    Parameters
    ----------
    r  : non-dimensional orbital radius from Earth
    mu : mass parameter

    Returns
    -------
    v_circ : non-dimensional speed
    """
    return np.sqrt((1.0 - mu) / r)


def leo_to_manifold_dv(
    manifold_insert_state: np.ndarray,
    r_parking: float = 6571.0 / L_STAR,   # ~200 km LEO in nd units
    mu: float = MU_EARTH_MOON,
) -> dict:
    """Estimate the ΔV from a circular LEO parking orbit to a manifold
    insertion point using a Hohmann-like two-impulse approach.

    This is an approximation: it assumes the insertion point is at a known
    radial distance from Earth and that the transfer is two-body dominated.
    A full CR3BP targeting would refine this with a shooting method.

    Parameters
    ----------
    manifold_insert_state : (6,) state at the manifold tube insertion point
    r_parking             : parking orbit radius from Earth (non-dim)
    mu                    : mass parameter

    Returns
    -------
    dict with keys:
        'r_insert'    : radial distance from Earth at insertion
        'v_circ_park' : circular speed at parking orbit
        'v_circ_ins'  : two-body circular speed at insertion radius
        'dv1_km_s'    : first impulse (perigee raise) in km/s
        'dv2_km_s'    : second impulse (circularise at insertion) in km/s
        'dv_total_km_s': total ΔV in km/s
    """
    x_i, y_i, z_i = manifold_insert_state[:3]
    # Distance from Earth (located at −μ on x-axis)
    r_insert = np.sqrt((x_i + mu) ** 2 + y_i ** 2 + z_i ** 2)

    v_circ_park = circular_orbit_velocity(r_parking, mu)
    v_circ_ins = circular_orbit_velocity(r_insert, mu)

    # Hohmann semi-major axis
    a_transfer = 0.5 * (r_parking + r_insert)
    v_perigee = np.sqrt((1.0 - mu) * (2.0 / r_parking - 1.0 / a_transfer))
    v_apogee  = np.sqrt((1.0 - mu) * (2.0 / r_insert  - 1.0 / a_transfer))

    dv1 = abs(v_perigee - v_circ_park)
    dv2 = abs(v_circ_ins - v_apogee)

    return {
        "r_insert_nd": r_insert,
        "r_insert_km": r_insert * L_STAR,
        "v_circ_park": v_circ_park,
        "v_circ_ins": v_circ_ins,
        "dv1_nd": dv1,
        "dv2_nd": dv2,
        "dv1_km_s": dv1 * V_STAR,
        "dv2_km_s": dv2 * V_STAR,
        "dv_total_km_s": (dv1 + dv2) * V_STAR,
    }


# ── Manifold-stitching ΔV ──────────────────────────────────────────────────────


def stitch_manifolds(
    state_dep: np.ndarray,
    state_arr: np.ndarray,
) -> dict:
    """Compute the ΔV needed to stitch two manifold arcs together.

    At the Poincaré section the spacecraft is on the departure manifold with
    velocity state_dep[3:] and needs to be on the arrival manifold with
    velocity state_arr[3:].  The impulsive ΔV is simply the velocity difference.

    Parameters
    ----------
    state_dep : (6,) state at the Poincaré section on the unstable manifold
    state_arr : (6,) state at the Poincaré section on the stable manifold

    Returns
    -------
    dict with keys:
        'delta_v_nd'  : (3,) ΔV vector in non-dimensional units
        'dv_norm_nd'  : |ΔV| non-dimensional
        'dv_km_s'     : |ΔV| in km/s
        'state_dep'   : input state_dep
        'state_arr'   : input state_arr
        'pos_error_nd': position mismatch at the section (should be small)
        'pos_error_km': position mismatch in km
    """
    dv = state_arr[3:] - state_dep[3:]
    dv_norm = np.linalg.norm(dv)
    pos_err = np.linalg.norm(state_arr[:3] - state_dep[:3])
    return {
        "delta_v_nd": dv,
        "dv_norm_nd": dv_norm,
        "dv_km_s": dv_norm * V_STAR,
        "state_dep": state_dep,
        "state_arr": state_arr,
        "pos_error_nd": pos_err,
        "pos_error_km": pos_err * L_STAR,
    }


# ── Full transfer ΔV budget ────────────────────────────────────────────────────


def full_transfer_budget(
    stitch_result: dict,
    parking_orbit_km: float = 200.0,
    mu: float = MU_EARTH_MOON,
) -> dict:
    """Assemble the full ΔV budget for a manifold-assisted transfer.

    Legs:
      1. TLI (Trans-Lunar Injection): LEO → manifold insertion  (Hohmann approx)
      2. Manifold coast: free (ballistic)
      3. Poincaré patch: ΔV to join arrival manifold
      4. Manifold coast: free (ballistic)
      5. LOI (Lunar Orbit Insertion): arrival orbit capture ≈ 0 for low-energy

    Parameters
    ----------
    stitch_result     : dict from stitch_manifolds()
    parking_orbit_km  : LEO altitude above Earth's surface (km)
    mu                : mass parameter

    Returns
    -------
    dict with full ΔV breakdown
    """
    R_earth_km = 6371.0
    r_parking_km = R_earth_km + parking_orbit_km
    r_parking_nd = r_parking_km / L_STAR

    dep_state = stitch_result["state_dep"]
    dep_budget = leo_to_manifold_dv(dep_state, r_parking=r_parking_nd, mu=mu)

    dv_patch_km_s = stitch_result["dv_km_s"]

    total = dep_budget["dv_total_km_s"] + dv_patch_km_s

    return {
        "leg1_TLI_km_s": dep_budget["dv_total_km_s"],
        "leg1_dv1_km_s": dep_budget["dv1_km_s"],
        "leg1_dv2_km_s": dep_budget["dv2_km_s"],
        "leg1_r_insert_km": dep_budget["r_insert_km"],
        "leg2_poincare_patch_km_s": dv_patch_km_s,
        "total_dv_km_s": total,
        "parking_orbit_km": parking_orbit_km,
        "parking_radius_km": r_parking_km,
        "pos_error_at_section_km": stitch_result["pos_error_km"],
    }


# ── Optional: low-thrust leg (tangential continuous thrust) ───────────────────


def low_thrust_eom(
    t: float,
    state: np.ndarray,
    mu: float,
    thrust_acc: float,
) -> np.ndarray:
    """CR3BP EOM with a constant tangential thrust acceleration.

    The thrust is applied in the direction of the velocity vector in the
    rotating frame, scaled to the given acceleration magnitude.

    Parameters
    ----------
    t           : time
    state       : (6,) [x, y, z, vx, vy, vz]
    mu          : mass parameter
    thrust_acc  : thrust acceleration magnitude (non-dimensional)

    Returns
    -------
    dstate/dt : (6,) time derivative
    """
    from .dynamics import cr3bp_eom as _eom
    deriv = _eom(t, state, mu).copy()
    vel = state[3:]
    v_norm = np.linalg.norm(vel)
    if v_norm > 1e-14:
        thrust_dir = vel / v_norm
        deriv[3:] += thrust_acc * thrust_dir
    return deriv


def propagate_low_thrust(
    state0: np.ndarray,
    t_span: tuple[float, float],
    thrust_acc: float,
    mu: float = MU_EARTH_MOON,
    t_eval: np.ndarray | None = None,
) -> object:
    """Propagate the CR3BP equations with constant tangential low thrust.

    Parameters
    ----------
    state0      : (6,) initial state
    t_span      : (t0, tf)
    thrust_acc  : non-dimensional thrust acceleration
                  (characteristic acceleration for a sail: β * (1-μ)/r²;
                   for a chemical thruster: T/(m * n² * L*))
    mu          : mass parameter
    t_eval      : optional output times

    Returns
    -------
    OdeResult from scipy.integrate.solve_ivp
    """
    return solve_ivp(
        low_thrust_eom,
        t_span,
        state0,
        args=(mu, thrust_acc),
        method="DOP853",
        rtol=1e-11,
        atol=1e-11,
        t_eval=t_eval,
        dense_output=True,
    )


def shoot_low_thrust_transfer(
    state_start: np.ndarray,
    state_target: np.ndarray,
    T_guess: float,
    thrust_acc: float,
    mu: float = MU_EARTH_MOON,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> dict:
    """Simple single-shooting to connect state_start to state_target via
    constant tangential low thrust.

    Free variables: initial velocity [vx0, vy0, vz0] and transfer time T.
    Constraints: final position and velocity match state_target.

    This is a ballpark targeting method — it does not optimize the thrust
    direction for fuel efficiency.

    Parameters
    ----------
    state_start  : (6,) departure state (position + initial velocity guess)
    state_target : (6,) target state at arrival
    T_guess      : initial transfer time guess
    thrust_acc   : constant thrust acceleration (nd)
    mu           : mass parameter
    tol          : convergence tolerance
    max_iter     : maximum Newton iterations

    Returns
    -------
    dict with keys:
        'converged'  : bool
        'state0'     : corrected departure state
        'T'          : transfer time
        'dv_km_s'    : velocity change at departure relative to state_start
        'trajectory' : (n_pts, 6) array of the transfer trajectory
    """
    x_vec = np.array([state_start[3], state_start[4], state_start[5], T_guess])
    pos_start = state_start[:3]
    pos_target = state_target[:3]
    vel_target = state_target[3:]

    converged = False
    for iteration in range(max_iter):
        vx0, vy0, vz0, T = x_vec
        s0 = np.concatenate([pos_start, [vx0, vy0, vz0]])
        sol = propagate_low_thrust(s0, (0.0, T), thrust_acc, mu)
        s_f = sol.y[:, -1]

        residual = np.concatenate([s_f[:3] - pos_target, s_f[3:] - vel_target])
        if np.linalg.norm(residual) < tol:
            converged = True
            break

        # Numerical Jacobian (finite differences)
        eps = 1e-7
        J = np.zeros((6, 4))
        for k in range(4):
            dx = x_vec.copy()
            dx[k] += eps
            vx_, vy_, vz_, T_ = dx
            s0_ = np.concatenate([pos_start, [vx_, vy_, vz_]])
            sol_ = propagate_low_thrust(s0_, (0.0, T_), thrust_acc, mu)
            s_f_ = sol_.y[:, -1]
            J[:, k] = (np.concatenate([s_f_[:3] - pos_target, s_f_[3:] - vel_target]) - residual) / eps

        # Newton update
        delta = -np.linalg.lstsq(J, residual, rcond=None)[0]
        x_vec += delta

    # Final trajectory
    vx0, vy0, vz0, T = x_vec
    s0_final = np.concatenate([pos_start, [vx0, vy0, vz0]])
    t_eval = np.linspace(0.0, T, 500)
    sol_final = propagate_low_thrust(s0_final, (0.0, T), thrust_acc, mu, t_eval=t_eval)

    dv = np.linalg.norm(s0_final[3:] - state_start[3:]) * V_STAR

    return {
        "converged": converged,
        "state0": s0_final,
        "T": T,
        "dv_km_s": dv,
        "trajectory": sol_final.y.T,
    }
