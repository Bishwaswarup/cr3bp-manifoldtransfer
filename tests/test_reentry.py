"""Tests for the Earth-approach / entry-interface module."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dynamics import L_STAR, MU_EARTH_MOON as MU, V_STAR  # noqa: E402
from src.reentry import (DAY_ND, EARTH_RADIUS_KM, earth_inertial_state, entry_conditions,  # noqa: E402
                         in_entry_corridor, propagate_to_earth)


def _state_from_inertial(r_km, v_km_s):
    """Rotating-frame state at t = 0 from an Earth-centred inertial state."""
    r = np.asarray(r_km) / L_STAR
    v = np.asarray(v_km_s) / V_STAR
    pos = r + np.array([-MU, 0, 0])
    v_earth = np.cross([0, 0, 1], [-MU, 0, 0])
    return np.concatenate([pos, v + v_earth - np.cross([0, 0, 1], pos)])


def test_inertial_round_trip():
    s = _state_from_inertial([7000, 1000, 500], [1.0, 7.0, 0.5])
    si = earth_inertial_state(s, 0.0, MU)
    assert np.allclose(si[:3] * L_STAR, [7000, 1000, 500], atol=1e-6)
    assert np.allclose(si[3:] * V_STAR, [1.0, 7.0, 0.5], atol=1e-9)


def test_flight_path_angle_of_known_state():
    # velocity tilted 6 degrees below the local horizontal, radial position on x
    g = np.radians(-6.0)
    s = _state_from_inertial([6498.137, 0, 0], [11.0 * np.sin(g), 11.0 * np.cos(g), 0])
    e = entry_conditions(s, 0.0, MU)
    assert abs(e["fpa_deg"] + 6.0) < 1e-6
    assert abs(e["speed_km_s"] - 11.0) < 1e-6
    assert abs(e["altitude_km"] - 120.0) < 1e-6
    assert in_entry_corridor(e["fpa_deg"])


def test_ballistic_entry_detected():
    # steep inbound trajectory from 50 000 km must cross the entry interface
    s = _state_from_inertial([50_000, 0, 0], [-3.0, 0.3, 0.0])
    r = propagate_to_earth(s, 5 * DAY_ND, MU)
    assert r["outcome"] == "entry"
    assert abs(r["entry"]["altitude_km"] - 120.0) < 1e-3
    assert r["entry"]["fpa_deg"] < -10


def test_perigee_recorded_without_entry():
    # near-circular 20 000 km orbit: no entry, perigees ~ 13 600 km altitude
    rr = 20_000.0
    vc = np.sqrt(398600.4418 / rr)
    s = _state_from_inertial([rr, 0, 0], [0, vc, 0])
    r = propagate_to_earth(s, 2 * DAY_ND, MU)
    assert r["outcome"] == "none"
    assert len(r["perigees"]) >= 1
    assert abs(r["min_perigee_alt_km"] - (rr - EARTH_RADIUS_KM)) < 500
