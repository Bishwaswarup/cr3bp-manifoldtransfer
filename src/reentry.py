"""
reentry.py
==========
Earth-approach map for the Earth-bound branch of a libration-point orbit's
unstable manifold (merged from the former ``space_highway`` repository).

For every sample point on the orbit the manifold trajectory is propagated
until it

  * crosses the atmospheric entry interface (Earth radius + 120 km),
  * impacts the Moon,
  * escapes the Earth–Moon system (r > 3 from Earth), or
  * reaches the time limit,

and every perigee passage on the way is recorded. At an entry-interface
crossing the inertial entry speed, the inertial flight-path angle and the
latitude/longitude of the entry point are returned.

Frames and conventions
----------------------
* The rotating frame and the Earth-centred inertial frame coincide at t = 0;
  the inertial frame rotates by -t relative to the rotating one.
* Latitude is measured from the Earth–Moon orbital plane, longitude from the
  Earth–Moon line at t = 0. They are *not* geographic coordinates: the CR3BP
  carries no information about Earth's spin axis or the epoch, so mapping to
  geographic latitude/longitude would need an ephemeris model.
* The flight-path angle uses the inertial velocity v_I = v_rot + ω × r.

Default entry corridor: flight-path angle between -7° and -5°, typical of
direct lunar-return entries (Apollo targeted about -6.5° at 122 km).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import (
    L_STAR,
    MU_EARTH_MOON,
    T_STAR,
    V_STAR,
    cr3bp_eom,
    cr3bp_eom_stm,
)
from .manifolds import stable_unstable_eigenvectors

EARTH_RADIUS_KM: float = 6378.137
MOON_RADIUS_KM: float = 1737.4
ENTRY_ALTITUDE_KM: float = 120.0
DAY_ND: float = 86400.0 / T_STAR          # one day in non-dimensional time


def _earth(mu: float) -> np.ndarray:
    return np.array([-mu, 0.0, 0.0])


def _moon(mu: float) -> np.ndarray:
    return np.array([1.0 - mu, 0.0, 0.0])


# ── Frame conversion and entry conditions ────────────────────────────────────


def earth_inertial_state(state: np.ndarray, t: float, mu: float = MU_EARTH_MOON) -> np.ndarray:
    """Earth-centred inertial state from a rotating-frame state at time t."""
    r = state[:3] - _earth(mu)
    v = state[3:] + np.cross([0.0, 0.0, 1.0], state[:3])     # ω × r about the barycentre
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # translate velocity of Earth's centre (it moves in the inertial frame too)
    v_earth = np.cross([0.0, 0.0, 1.0], _earth(mu))
    return np.concatenate([R @ r, R @ (v - v_earth)])


def entry_conditions(state: np.ndarray, t: float, mu: float = MU_EARTH_MOON) -> dict:
    """Inertial entry speed, flight-path angle and entry-point coordinates."""
    s_i = earth_inertial_state(state, t, mu)
    r, v = s_i[:3], s_i[3:]
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    return {
        "t_days": t / DAY_ND,
        "altitude_km": rn * L_STAR - EARTH_RADIUS_KM,
        "speed_km_s": vn * V_STAR,
        "fpa_deg": float(np.degrees(np.arcsin(np.dot(r, v) / (rn * vn)))),
        "lat_deg": float(np.degrees(np.arcsin(r[2] / rn))),
        "lon_deg": float(np.degrees(np.arctan2(r[1], r[0]))),
    }


def in_entry_corridor(fpa_deg: float, lo: float = -7.0, hi: float = -5.0) -> bool:
    """True if the flight-path angle lies inside the entry corridor."""
    return lo <= fpa_deg <= hi


# ── Single trajectory ─────────────────────────────────────────────────────────


def propagate_to_earth(
    state: np.ndarray,
    t_max: float,
    mu: float = MU_EARTH_MOON,
    entry_altitude_km: float = ENTRY_ALTITUDE_KM,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> dict:
    """Propagate one trajectory and classify how it ends.

    Returns
    -------
    dict with keys
        'outcome'           : 'entry' | 'moon' | 'escape' | 'none'
        't_end_days'        : time at which propagation stopped
        'perigees'          : (N, 2) array of (t_days, altitude_km) at each perigee
        'min_perigee_alt_km': lowest perigee altitude (entry altitude if it entered)
        'entry'             : entry_conditions(...) dict, or None
    """
    r_ei = (EARTH_RADIUS_KM + entry_altitude_km) / L_STAR
    r_moon = MOON_RADIUS_KM / L_STAR
    E, M = _earth(mu), _moon(mu)

    def ev_entry(t, s, mu): return np.linalg.norm(s[:3] - E) - r_ei
    ev_entry.terminal, ev_entry.direction = True, -1

    def ev_moon(t, s, mu): return np.linalg.norm(s[:3] - M) - r_moon
    ev_moon.terminal, ev_moon.direction = True, -1

    def ev_escape(t, s, mu): return np.linalg.norm(s[:3] - E) - 3.0
    ev_escape.terminal, ev_escape.direction = True, 1

    def ev_perigee(t, s, mu): return np.dot(s[:3] - E, s[3:])
    ev_perigee.terminal, ev_perigee.direction = False, 1

    sol = solve_ivp(cr3bp_eom, (0.0, t_max), state, args=(mu,), method="DOP853",
                    rtol=rtol, atol=atol,
                    events=[ev_entry, ev_moon, ev_escape, ev_perigee])

    per = np.array([[t / DAY_ND, np.linalg.norm(s[:3] - E) * L_STAR - EARTH_RADIUS_KM]
                    for t, s in zip(sol.t_events[3], sol.y_events[3])]).reshape(-1, 2)
    outcome, entry = "none", None
    if sol.t_events[0].size:
        outcome = "entry"
        entry = entry_conditions(sol.y_events[0][0], sol.t_events[0][0], mu)
    elif sol.t_events[1].size:
        outcome = "moon"
    elif sol.t_events[2].size:
        outcome = "escape"

    min_alt = entry_altitude_km if outcome == "entry" else (per[:, 1].min() if per.size else np.inf)
    return {
        "outcome": outcome,
        "t_end_days": sol.t[-1] / DAY_ND,
        "perigees": per,
        "min_perigee_alt_km": float(min_alt),
        "entry": entry,
    }


# ── Manifold map ──────────────────────────────────────────────────────────────


def earth_approach_map(
    state0: np.ndarray,
    T: float,
    mu: float = MU_EARTH_MOON,
    n_samples: int = 60,
    eps: float = 1e-6,
    t_max_days: float = 180.0,
    direction: int | None = None,
    entry_altitude_km: float = ENTRY_ALTITUDE_KM,
) -> list[dict]:
    """Propagate the interior (Earth-bound) unstable manifold from n_samples
    points around a periodic orbit and record each trajectory's fate.

    Parameters
    ----------
    state0, T  : periodic orbit (initial state and full period)
    n_samples  : number of points sampled uniformly in time around the orbit
    eps        : perturbation along the local unstable eigenvector (nd)
    t_max_days : propagation limit
    direction  : +1 / -1 to choose the branch; None picks the interior branch
                 (initial perturbation towards -x, i.e. towards Earth)

    Returns
    -------
    list of dicts, one per sample, with 'tau' (orbit phase in [0, 1)) plus
    the keys of :func:`propagate_to_earth`.
    """
    t_s = np.linspace(0.0, T, n_samples, endpoint=False)
    y0 = np.concatenate([state0, np.eye(6).ravel()])
    sol = solve_ivp(cr3bp_eom_stm, (0.0, T), y0, args=(mu,), method="DOP853",
                    rtol=1e-12, atol=1e-12, t_eval=t_s)

    M = sol.y[6:, -1].reshape(6, 6) if np.isclose(sol.t[-1], T) else None
    if M is None:
        _ = solve_ivp(cr3bp_eom_stm, (0.0, T), y0, args=(mu,), method="DOP853",
                      rtol=1e-12, atol=1e-12)
        M = _.y[6:, -1].reshape(6, 6)
    v_u, _ = stable_unstable_eigenvectors(M)
    if direction is None:
        direction = -1 if v_u[0] > 0 else 1

    out = []
    for k, t_k in enumerate(sol.t):
        s_k = sol.y[:6, k]
        v_k = sol.y[6:, k].reshape(6, 6) @ v_u
        v_k /= np.linalg.norm(v_k)
        res = propagate_to_earth(s_k + direction * eps * v_k, t_max_days * DAY_ND, mu,
                                 entry_altitude_km=entry_altitude_km)
        res["tau"] = t_k / T
        out.append(res)
    return out


def summarize_map(results: list[dict]) -> dict:
    """Counts by outcome and the lowest perigee over a map."""
    outcomes = [r["outcome"] for r in results]
    alts = np.array([r["min_perigee_alt_km"] for r in results])
    entries = [r["entry"] for r in results if r["entry"] is not None]
    return {
        "n": len(results),
        "entry": outcomes.count("entry"),
        "moon": outcomes.count("moon"),
        "escape": outcomes.count("escape"),
        "none": outcomes.count("none"),
        "min_perigee_alt_km": float(np.min(alts)) if alts.size else np.inf,
        "median_perigee_alt_km": float(np.median(alts[np.isfinite(alts)])) if np.isfinite(alts).any() else np.inf,
        "in_corridor": sum(in_entry_corridor(e["fpa_deg"]) for e in entries),
    }
