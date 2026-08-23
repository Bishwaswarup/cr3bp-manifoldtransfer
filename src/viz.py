"""
viz.py
======
Visualization routines for the CR3BP manifold transfer pipeline.

Provides:
  - 3D manifold tube plots in the rotating frame
  - Poincaré section scatter plots
  - ΔV budget summary table
  - Optional interactive Plotly variants

All plot functions return the matplotlib Figure so the caller can save or
show them. Call plt.show() or fig.savefig(…) externally.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.gridspec import GridSpec

from .dynamics import MU_EARTH_MOON, L_STAR, lagrange_points

# ── Styling ────────────────────────────────────────────────────────────────────

COLORS = {
    "unstable": "#e05c5c",   # red
    "stable":   "#5c8ae0",   # blue
    "orbit_dep": "#f0a020",  # amber  (departure halo)
    "orbit_arr": "#20c090",  # teal   (arrival halo)
    "transfer":  "#c060e0",  # purple
    "primaries": ["#1a1a2e", "#778ca3"],
    "section":   "#aaaaaa",
}


def _primary_positions(mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Earth and Moon positions in the rotating frame."""
    earth = np.array([-mu, 0.0, 0.0])
    moon  = np.array([1.0 - mu, 0.0, 0.0])
    return earth, moon


def plot_manifold_tubes_3d(
    orbit_dep: np.ndarray,
    manifold_dep: list[np.ndarray],
    orbit_arr: np.ndarray | None = None,
    manifold_arr: list[np.ndarray] | None = None,
    transfer_traj: np.ndarray | None = None,
    mu: float = MU_EARTH_MOON,
    n_tubes: int = 40,
    figsize: tuple = (11, 8),
    title: str = "CR3BP Manifold-Assisted Transfer",
) -> plt.Figure:
    """3D rotating-frame plot of manifold tubes and the transfer trajectory.

    Parameters
    ----------
    orbit_dep     : (n_pts, 6) departure periodic orbit
    manifold_dep  : list of (n_pts, 6) unstable manifold trajectories
    orbit_arr     : (n_pts, 6) arrival periodic orbit (optional)
    manifold_arr  : list of (n_pts, 6) stable manifold trajectories (optional)
    transfer_traj : (n_pts, 6) low-thrust transfer leg (optional)
    mu            : mass parameter
    n_tubes       : number of manifold tubes to draw (subsample for clarity)
    figsize       : figure size
    title         : plot title

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    earth, moon = _primary_positions(mu)

    # Primary bodies
    ax.scatter(*earth, s=80, color=COLORS["primaries"][0], zorder=10,
               label="Earth", marker="o")
    ax.scatter(*moon,  s=40, color=COLORS["primaries"][1], zorder=10,
               label="Moon",  marker="o")

    # Departure orbit
    ax.plot(orbit_dep[:, 0], orbit_dep[:, 1], orbit_dep[:, 2],
            color=COLORS["orbit_dep"], lw=2.0, label="Departure orbit", zorder=5)

    # Arrival orbit
    if orbit_arr is not None:
        ax.plot(orbit_arr[:, 0], orbit_arr[:, 1], orbit_arr[:, 2],
                color=COLORS["orbit_arr"], lw=2.0, label="Arrival orbit", zorder=5)

    # Unstable manifold tubes (departure)
    stride = max(1, len(manifold_dep) // n_tubes)
    for i, traj in enumerate(manifold_dep[::stride]):
        lbl = "Unstable manifold" if i == 0 else "_"
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=COLORS["unstable"], lw=0.5, alpha=0.4, label=lbl)

    # Stable manifold tubes (arrival)
    if manifold_arr is not None:
        stride_a = max(1, len(manifold_arr) // n_tubes)
        for i, traj in enumerate(manifold_arr[::stride_a]):
            lbl = "Stable manifold" if i == 0 else "_"
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=COLORS["stable"], lw=0.5, alpha=0.4, label=lbl)

    # Low-thrust leg
    if transfer_traj is not None:
        ax.plot(transfer_traj[:, 0], transfer_traj[:, 1], transfer_traj[:, 2],
                color=COLORS["transfer"], lw=2.0, ls="--", label="Low-thrust arc", zorder=6)

    ax.set_xlabel("x (nd)", labelpad=6)
    ax.set_ylabel("y (nd)", labelpad=6)
    ax.set_zlabel("z (nd)", labelpad=6)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    return fig


def plot_poincare_section(
    crossings_dep: np.ndarray,
    crossings_arr: np.ndarray | None = None,
    best_match: dict | None = None,
    coords: tuple[int, int] = (0, 5),  # (x-index, y-index) in state vector
    xlabel: str = "x (nd)",
    ylabel: str = "vy (nd)",
    title: str = "Poincaré Section (y = 0)",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Poincaré section scatter plot showing manifold intersections.

    Parameters
    ----------
    crossings_dep : (N1, 6) unstable manifold crossings
    crossings_arr : (N2, 6) stable manifold crossings (optional)
    best_match    : dict from find_manifold_intersections (optional)
    coords        : which state indices to plot as (horizontal, vertical)
    xlabel/ylabel : axis labels
    title         : plot title
    figsize       : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f8f9fa")

    ix, iy = coords

    if crossings_dep.shape[0] > 0:
        ax.scatter(crossings_dep[:, ix], crossings_dep[:, iy],
                   s=20, color=COLORS["unstable"], alpha=0.7,
                   label="Unstable manifold (departure)", zorder=4)

    if crossings_arr is not None and crossings_arr.shape[0] > 0:
        ax.scatter(crossings_arr[:, ix], crossings_arr[:, iy],
                   s=20, color=COLORS["stable"], alpha=0.7,
                   label="Stable manifold (arrival)", zorder=4)

    if best_match is not None:
        sd = best_match["state_dep"]
        sa = best_match["state_arr"]
        ax.scatter([sd[ix]], [sd[iy]], s=120, color=COLORS["unstable"],
                   edgecolors="k", lw=1.5, zorder=8, label="Best match (dep)")
        ax.scatter([sa[ix]], [sa[iy]], s=120, color=COLORS["stable"],
                   edgecolors="k", lw=1.5, zorder=8, label="Best match (arr)")
        ax.annotate(
            f"ΔV = {best_match['dv_km_s']:.3f} km/s",
            xy=(0.5 * (sd[ix] + sa[ix]), 0.5 * (sd[iy] + sa[iy])),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#555"),
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig


def print_dv_budget(budget: dict) -> None:
    """Print a formatted ΔV budget table to stdout.

    Parameters
    ----------
    budget : dict from full_transfer_budget()
    """
    sep = "─" * 52
    print()
    print("  CR3BP Manifold Transfer — ΔV Budget")
    print(sep)
    print(f"  Parking orbit altitude       : {budget['parking_orbit_km']:.0f} km LEO")
    print(f"  Parking orbit radius         : {budget['parking_radius_km']:.0f} km")
    print(f"  Manifold insertion radius    : {budget['leg1_r_insert_km']:.0f} km")
    print(sep)
    print(f"  Leg 1a — TLI perigee burn    : {budget['leg1_dv1_km_s']:.4f} km/s")
    print(f"  Leg 1b — insertion burn      : {budget['leg1_dv2_km_s']:.4f} km/s")
    print(f"  Leg 2  — Poincaré patch ΔV   : {budget['leg2_poincare_patch_km_s']:.4f} km/s")
    print(sep)
    print(f"  TOTAL ΔV                     : {budget['total_dv_km_s']:.4f} km/s")
    print(f"  Position error at section    : {budget['pos_error_at_section_km']:.2f} km")
    print(sep)
    print()


def plot_dv_budget_bar(budget: dict, figsize: tuple = (7, 4)) -> plt.Figure:
    """Bar chart of the ΔV breakdown.

    Parameters
    ----------
    budget  : dict from full_transfer_budget()
    figsize : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    labels = ["TLI perigee\nburn", "Manifold\ninsertion", "Poincaré\npatch ΔV"]
    values = [
        budget["leg1_dv1_km_s"],
        budget["leg1_dv2_km_s"],
        budget["leg2_poincare_patch_km_s"],
    ]
    colors = ["#5c8ae0", "#5caacc", "#e05c5c"]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.axhline(budget["total_dv_km_s"], color="#333", ls="--", lw=1.5,
               label=f"Total ΔV = {budget['total_dv_km_s']:.3f} km/s")
    ax.set_ylabel("ΔV (km/s)", fontsize=11)
    ax.set_title("Transfer ΔV Budget", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig
