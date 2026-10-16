"""Regenerate every figure in figures/ and the numbers in data/results.json.

    python scripts/make_figures.py            # full run, ~5 min on a laptop
    python scripts/make_figures.py --quick    # fewer samples, ~1 min

All numbers quoted in the README come from data/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dynamics import L_STAR, MU_EARTH_MOON as MU, T_STAR, lagrange_points, propagate  # noqa: E402
from src.orbits import (compute_halo_family, differential_correction, periodicity_error,  # noqa: E402
                        richardson_guess, stability_index, z_amplitude)
from src.connections import energy_matched_connection, manifold_section_points  # noqa: E402
from src.reentry import (DAY_ND, EARTH_RADIUS_KM, earth_approach_map, propagate_to_earth,  # noqa: E402
                         summarize_map)

FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DAYS = T_STAR / 86400.0

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 1.6,
    "legend.frameon": False,
})
KM = L_STAR / 1000.0  # plot positions in 10^3 km


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  saved figures/" + name)


def orbit_xyz(state0, T, n=600):
    sol = propagate(state0, (0.0, T), mu=MU, t_eval=np.linspace(0.0, T, n))
    return sol.y[:3]


def primaries(ax, dims=(0, 1), moon=True, earth=True):
    pts = {"Earth": (np.array([-MU, 0, 0]), "#3b6ea8", 40), "Moon": (np.array([1 - MU, 0, 0]), "#8c8b86", 22)}
    for name, (p, c, s) in pts.items():
        if (name == "Moon" and moon) or (name == "Earth" and earth):
            ax.scatter(p[dims[0]] * KM, p[dims[1]] * KM, s=s, c=c, zorder=5)
            ax.annotate(name, (p[dims[0]] * KM, p[dims[1]] * KM), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color=MUTED)


def main(quick: bool):
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    res: dict = {}
    t0 = time.time()

    # ── 1. Reference halo orbit ──────────────────────────────────────────────
    print("1/5 reference L1 halo (Az = 0.05)")
    guess, _ = richardson_guess(0.05, "L1", MU)
    s0, th = differential_correction(guess, mu=MU)
    T = 2 * th
    res["reference_halo"] = {
        "Az_nd_input": 0.05,
        "guess": guess.tolist(), "state0": s0.tolist(),
        "guess_error_km": float(np.linalg.norm(guess[:3] - s0[:3]) * L_STAR),
        "period_days": T * DAYS, "z_amplitude_km": z_amplitude(s0, T, MU) * L_STAR,
        "jacobi": float(__import__("src.dynamics", fromlist=["x"]).jacobi_constant(s0, MU)),
        "stability_index": stability_index(s0, T, MU),
        "periodicity_error_nd": periodicity_error(s0, T, MU),
    }
    xyz = orbit_xyz(s0, T)
    xL1 = lagrange_points(MU)["L1"][0]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (i, j), lab in zip(axs, [(0, 1), (0, 2), (1, 2)], ["xy", "xz", "yz"]):
        ax.plot(xyz[i] * KM, xyz[j] * KM, color=BLUE)
        ax.scatter(s0[i] * KM, s0[j] * KM, s=30, color=INK, zorder=5, label="x₀")
        if i == 0:
            ax.axvline(xL1 * KM, color=MUTED, lw=0.8, ls="--")
            ax.annotate("L1", (xL1 * KM, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points",
                        fontsize=8, color=MUTED)
        ax.set_xlabel(f"{'xyz'[i]} (10³ km)")
        ax.set_ylabel(f"{'xyz'[j]} (10³ km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{lab} projection")
    fig.suptitle(f"Earth–Moon L1 northern halo: A_z = {res['reference_halo']['z_amplitude_km']:,.0f} km, "
                 f"T = {T * DAYS:.2f} d, periodicity error {res['reference_halo']['periodicity_error_nd']:.1e}",
                 fontsize=11, y=1.04)
    save(fig, "halo_orbit.png")

    # ── 2. Families ──────────────────────────────────────────────────────────
    print("2/5 L1 and L2 halo families (pseudo-arclength continuation)")
    n_fam = 25 if quick else 50
    f1 = compute_halo_family(0.005, 0.20, n_orbits=n_fam, lp="L1", ds=0.005)
    f2 = compute_halo_family(0.005, 0.20, n_orbits=n_fam, lp="L2", ds=0.005)
    per_err = max(periodicity_error(m["state0"], m["T"], MU) for m in f1 + f2)
    res["families"] = {
        lp: {"n": len(f), "Az_km": [m["Az"] * L_STAR for m in f], "period_days": [m["T"] * DAYS for m in f],
             "C": [m["C"] for m in f], "nu": [m["nu"] for m in f]}
        for lp, f in (("L1", f1), ("L2", f2))
    }
    res["families"]["max_periodicity_error_nd"] = per_err

    fig = plt.figure(figsize=(12, 4.2))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    for f, c in ((f1, BLUE), (f2, ORANGE)):
        for m in f[:: max(1, len(f) // 12)]:
            x, y, z = orbit_xyz(m["state0"], m["T"], 300) * KM
            ax3.plot(x, y, z, color=c, lw=0.8, alpha=0.85)
    ax3.scatter([(1 - MU) * KM], [0], [0], s=20, c="#8c8b86")
    ax3.set_xlabel("x (10³ km)"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title("Halo families (blue L1, orange L2)")
    ax3.view_init(elev=18, azim=-60)
    ax = fig.add_subplot(1, 3, 2)
    for f, c, lab in ((f1, BLUE, "L1"), (f2, ORANGE, "L2")):
        ax.plot([m["Az"] * L_STAR / 1e3 for m in f], [m["T"] * DAYS for m in f], "o-", ms=3, color=c, label=lab)
    ax.set_xlabel("out-of-plane amplitude A_z (10³ km)"); ax.set_ylabel("period (days)")
    ax.legend(); ax.set_title("Period along the family")
    ax = fig.add_subplot(1, 3, 3)
    for f, c, lab in ((f1, BLUE, "L1"), (f2, ORANGE, "L2")):
        ax.semilogy([m["Az"] * L_STAR / 1e3 for m in f], [m["nu"] for m in f], "o-", ms=3, color=c, label=lab)
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("A_z (10³ km)"); ax.set_ylabel("stability index ν")
    ax.legend(); ax.set_title("Stability (ν = 1: stable)")
    fig.tight_layout()
    save(fig, "halo_family.png")

    # ── 3. Energy-matched L1 → L2 connections ────────────────────────────────
    print("3/5 energy-matched L1/L2 pairs and section matching")
    Cs = [3.15, 3.13, 3.11, 3.09, 3.07] if not quick else [3.15, 3.11, 3.07]
    n_sec = 60 if quick else 150
    rows, ref = [], None
    for C in Cs:
        r = energy_matched_connection(C, f1, f2, MU, n_samples=n_sec)
        p = r["patch"]
        rows.append({"C": C, "Az_L1_km": r["Az_L1_km"], "Az_L2_km": r["Az_L2_km"],
                     "n_dep": len(r["section_dep"]), "n_arr": len(r["section_arr"]),
                     "min_gap_km": p["min_gap_km"], "dv_at_min_gap_m_s": p["dv_at_min_gap_km_s"] * 1e3,
                     "best_dv_m_s": p["dv_km_s"] * 1e3, "best_dv_gap_km": p["gap_km"]})
        print(f"    C={C}: Az {r['Az_L1_km']:.0f}/{r['Az_L2_km']:.0f} km, min gap {p['min_gap_km']:.0f} km, "
              f"best dv {p['dv_km_s'] * 1e3:.0f} m/s at gap {p['gap_km']:.0f} km")
        if C == 3.13 or (ref is None and C == Cs[len(Cs) // 2]):
            ref = r
    res["connections"] = rows

    # tubes figure for the reference energy
    s1, T1 = ref["orbit_L1"]
    s2, T2 = ref["orbit_L2"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    for (s, T_, branch, c) in ((s1, T1, "unstable", BLUE), (s2, T2, "stable", ORANGE)):
        _, arcs = manifold_section_points(s, T_, MU, branch=branch, n_samples=40, return_arcs=True)
        for arc in arcs:
            ax.plot(arc[0] * KM, arc[1] * KM, color=c, lw=0.5, alpha=0.6)
        x, y, _ = orbit_xyz(s, T_)
        ax.plot(x * KM, y * KM, color=INK, lw=1.4)
    ax.axvline((1 - MU) * KM, color=MUTED, lw=0.8, ls="--")
    primaries(ax, earth=False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (10³ km)"); ax.set_ylabel("y (10³ km)")
    ax.set_title(f"C = {ref['C']}: L1 unstable (blue) and L2 stable (orange) tubes")
    ax = axs[1]
    d, a = ref["section_dep"], ref["section_arr"]
    d, a = d[d[:, 3] > 0], a[a[:, 3] > 0]
    ax.plot(d[:, 1] * KM, d[:, 2] * KM, "o", ms=3.5, color=BLUE, label="W^u(L1 halo)")
    ax.plot(a[:, 1] * KM, a[:, 2] * KM, "o", ms=3.5, color=ORANGE, label="W^s(L2 halo)")
    p = ref["patch"]
    ax.plot([p["state_dep"][1] * KM, p["state_arr"][1] * KM], [p["state_dep"][2] * KM, p["state_arr"][2] * KM],
            color=INK, lw=1.2)
    ax.annotate(f"best patch: gap {p['gap_km']:.0f} km, ΔV {p['dv_km_s'] * 1e3:.0f} m/s",
                (p["state_dep"][1] * KM, p["state_dep"][2] * KM), xytext=(0.03, 0.95), textcoords="axes fraction",
                fontsize=9, color=INK, va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("y (10³ km)"); ax.set_ylabel("z (10³ km)")
    ax.set_title("First crossings of x = 1 − μ (through the Moon), ẋ > 0")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "manifold_tubes.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    Cv = [r["C"] for r in rows]
    axs[0].plot(Cv, [r["min_gap_km"] for r in rows], "o-", color=BLUE)
    axs[0].set_xlabel("Jacobi constant C"); axs[0].set_ylabel("smallest position gap (km)")
    axs[0].set_title("How close the two tubes come")
    axs[1].plot(Cv, [r["best_dv_m_s"] for r in rows], "o-", color=ORANGE)
    axs[1].set_xlabel("Jacobi constant C"); axs[1].set_ylabel("ΔV of best patch (m/s)")
    axs[1].set_title("Smallest ΔV with gap < 1000 km")
    for ax in axs:
        ax.invert_xaxis()
    fig.tight_layout()
    save(fig, "connection_dv.png")

    # ── 4. Earth-approach map (merged from space_highway) ────────────────────
    print("4/5 Earth-approach map of the interior L1 unstable manifold")
    n_map = 24 if quick else 60
    t_max = 180.0 if quick else 365.0
    picks = [m for m in f1 if 5e3 < m["Az"] * L_STAR < 70e3]
    picks = picks[:: max(1, len(picks) // (4 if quick else 8))]
    maps = []
    for m in picks:
        r = earth_approach_map(m["state0"], m["T"], MU, n_samples=n_map, t_max_days=t_max)
        sm = summarize_map(r)
        sm.update({"Az_km": m["Az"] * L_STAR, "C": m["C"], "nu": m["nu"]})
        maps.append((m, r, sm))
        print(f"    Az {sm['Az_km']:.0f} km: entry {sm['entry']}/{sm['n']}, moon {sm['moon']}, escape {sm['escape']}, "
              f"lowest perigee {sm['min_perigee_alt_km']:.0f} km")
    res["earth_approach"] = {"t_max_days": t_max, "n_samples": n_map, "entry_altitude_km": 120.0,
                             "orbits": [sm for _, _, sm in maps]}

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    m0, r0, sm0 = maps[len(maps) // 2]
    ax = axs[0]
    sol = propagate(m0["state0"], (0, m0["T"]), mu=MU, t_eval=np.linspace(0, m0["T"], 400))
    ax.plot(sol.y[0] * KM, sol.y[1] * KM, color=INK, lw=1.2)
    from src.manifolds import stable_unstable_eigenvectors
    from src.dynamics import propagate_stm
    _, Mm = propagate_stm(m0["state0"], (0, m0["T"]), MU)
    vu, _ = stable_unstable_eigenvectors(Mm)
    dirn = -1 if vu[0] > 0 else 1
    for tk in np.linspace(0, m0["T"], 10, endpoint=False):
        if tk > 0:
            so, P = propagate_stm(m0["state0"], (0, tk), MU)
            sk = so.y[:6, -1]
        else:
            sk, P = m0["state0"].copy(), np.eye(6)
        v = P @ vu
        v /= np.linalg.norm(v)
        tr = propagate(sk + dirn * 1e-6 * v, (0, 60 * DAY_ND), mu=MU, t_eval=np.linspace(0, 60 * DAY_ND, 3000))
        ax.plot(tr.y[0] * KM, tr.y[1] * KM, color=BLUE, lw=0.5, alpha=0.6)
    ax.add_patch(plt.Circle((-MU * KM, 0), (EARTH_RADIUS_KM + sm0["min_perigee_alt_km"]) / 1e3, fill=False,
                            ls="--", color=ORANGE, lw=1))
    primaries(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (10³ km)"); ax.set_ylabel("y (10³ km)")
    ax.set_title(f"Interior W^u, A_z = {sm0['Az_km'] / 1e3:.0f}×10³ km (60 d)\n"
                 "dashed: lowest perigee over the map", fontsize=10)
    ax = axs[1]
    for k, (m, r, sm) in enumerate(maps):
        col = plt.cm.viridis(k / max(1, len(maps) - 1))
        ax.plot([q["tau"] for q in r], [q["min_perigee_alt_km"] / 1e3 for q in r], "-", color=col, lw=1,
                label=f"{sm['Az_km'] / 1e3:.0f}")
    ax.axhline(0.12, color=ORANGE, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("departure phase on the halo τ"); ax.set_ylabel("lowest perigee altitude (10³ km)")
    ax.set_title(f"Lowest perigee within {t_max:.0f} d (dashed: 120 km entry interface)", fontsize=10)
    ax.legend(title="A_z (10³ km)", fontsize=7, title_fontsize=8, ncol=2)
    ax = axs[2]
    az = [sm["Az_km"] / 1e3 for _, _, sm in maps]
    bottom = np.zeros(len(maps))
    for key, c in (("none", "#b9b8b2"), ("escape", VIOLET), ("moon", AQUA), ("entry", ORANGE)):
        vals = np.array([sm[key] / sm["n"] for _, _, sm in maps])
        ax.bar(range(len(maps)), vals, bottom=bottom, color=c, label=key, edgecolor="#fcfcfb", linewidth=1)
        bottom += vals
    ax.set_xticks(range(len(maps)), [f"{a:.0f}" for a in az])
    ax.set_xlabel("A_z (10³ km)"); ax.set_ylabel("fraction of trajectories")
    ax.set_title(f"Fate within {t_max:.0f} d", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save(fig, "earth_approach_map.png")

    # ── 5. Write numbers ─────────────────────────────────────────────────────
    res["runtime_s"] = time.time() - t0
    res["quick"] = quick
    with open(os.path.join(DATA, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"5/5 wrote data/results.json ({res['runtime_s']:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
