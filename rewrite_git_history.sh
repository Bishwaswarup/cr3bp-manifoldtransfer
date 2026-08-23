#!/usr/bin/env bash
# =============================================================================
# rewrite_git_history.sh
# Run this ONCE from inside your cr3bp-manifold-transfer repo root to replace
# the single "initial commit" with a clean, logical commit history.
#
# Usage:
#   cd /path/to/cr3bp-manifold-transfer
#   bash rewrite_git_history.sh
#
# WARNING: This rewrites history. If the repo is already pushed to GitHub,
#          you will need to force-push afterwards:
#              git push --force origin main
# =============================================================================

set -e  # abort on any error

# ---- safety check -----------------------------------------------------------
if [ ! -f "src/dynamics.py" ]; then
  echo "ERROR: Run this script from the repo root (the folder containing src/)."
  exit 1
fi

REPO_ROOT="$(pwd)"

echo ">>> Backing up current branch tip..."
ORIG_HEAD=$(git rev-parse HEAD)
echo "    Original HEAD: $ORIG_HEAD"

# ---- 1. Detach HEAD and start a fresh orphan branch -------------------------
echo ""
echo ">>> Creating fresh orphan branch 'main-rewritten'..."
git checkout --orphan main-rewritten
git rm -rf --cached . > /dev/null   # unstage everything (files stay on disk)

# ---- Commit 1: project skeleton + CR3BP dynamics ----------------------------
echo ""
echo ">>> Commit 1/8 — project skeleton + CR3BP dynamics"
git add README.md LICENSE requirements.txt CITATION.cff .gitignore 2>/dev/null || true
git add src/__init__.py src/dynamics.py
git commit -m "feat: project skeleton and CR3BP dynamics core

Add non-dimensional CR3BP equations of motion, state-transition-matrix
(STM) augmented propagator, Jacobi constant, and Lagrange-point solver
(brentq root-finding). Constants: μ=0.01215, L*=384400 km, T*=375699.8 s."

# ---- Commit 2: halo orbit computation ---------------------------------------
echo ">>> Commit 2/8 — halo orbit computation"
git add src/orbits.py
git commit -m "feat: Richardson initial guess and differential correction for halo orbits

Implement 3rd-order Richardson (1980) analytic expansion for L1/L2 halo
guess.  Single-shooting differential corrector uses 2×3 Jacobian from STM
(free variables z0, ẏ0, T/2; constraints ẋ=ż=0 at half-period crossing)."

# ---- Commit 3: invariant manifolds ------------------------------------------
echo ">>> Commit 3/8 — invariant manifold tube generation"
git add src/manifolds.py
git commit -m "feat: monodromy matrix and invariant manifold tube generation

Compute monodromy M=Φ(T,0).  Extract stable (λ<1) and unstable (λ>1)
eigenvectors.  Perturb orbit by ε=1e-6 along eigenvector at n_manifold
equally-spaced points and integrate ±3T to form manifold tubes."

# ---- Commit 4: Poincaré section ---------------------------------------------
echo ">>> Commit 4/8 — Poincaré section and intersection search"
git add src/poincare.py
git commit -m "feat: Poincaré section (y=0) crossing detection and intersection search

Sign-change interpolation finds tube crossings on the y=0 plane.
Brute-force O(N1×N2) search minimises position mismatch; returns ranked
list of candidate transfers sorted by ΔV_norm."

# ---- Commit 5: transfer ΔV budget -------------------------------------------
echo ">>> Commit 5/8 — transfer ΔV budget and low-thrust leg"
git add src/transfer.py
git commit -m "feat: ΔV budget — TLI estimate, manifold stitch, and low-thrust leg

Hohmann TLI from 200 km LEO to manifold insertion.  Manifold-stitching ΔV
at Poincaré intersection.  Optional continuous tangential-thrust arc from
LEO targeted via single-shooting (numerical Jacobian)."

# ---- Commit 6: visualisation ------------------------------------------------
echo ">>> Commit 6/8 — visualisation utilities"
git add src/viz.py
git commit -m "feat: 3-D manifold tube plots, Poincaré scatter, ΔV bar chart

plot_manifold_tubes_3d — rotating-frame 3-D figure.
plot_poincare_section  — (x, ẋ) or (x, z) scatter of tube crossings.
plot_dv_budget_bar     — stacked bar with total-ΔV reference line."

# ---- Commit 7: demo notebook + generated figures ----------------------------
echo ">>> Commit 7/8 — demo notebook and generated figures"
git add notebooks/
git add figures/*.png 2>/dev/null || true
git commit -m "feat: end-to-end demo notebook with generated figures

Eight-cell Jupyter notebook runs the full pipeline: Lagrange points →
Richardson guess → differential correction → monodromy → manifold tubes →
Poincaré section → intersection search → ΔV budget → halo-family
continuation.  Generated figures committed for README preview."

# ---- Commit 8: paper + tests ------------------------------------------------
echo ">>> Commit 8/8 — LaTeX paper and unit tests"
git add paper/
git add tests/
git add data/.gitkeep 2>/dev/null || true
git commit -m "docs+test: journal-ready LaTeX paper and pytest unit-test suite

paper/main.tex  — Celestial Mechanics / Acta Astronautica style paper with
full derivations (CR3BP, halo correction, monodromy, manifolds, ΔV).
tests/test_dynamics.py — 10 tests: distances, Jacobi conservation,
STM volume preservation (det=1), Lagrange-point locations."

# ---- Replace main with rewritten branch ------------------------------------
echo ""
echo ">>> Replacing 'main' with rewritten history..."
git branch -D main 2>/dev/null || true
git branch -m main-rewritten main
git checkout main

echo ""
echo "================================================================="
echo " Done!  New commit log:"
echo "================================================================="
git log --oneline

echo ""
echo ">>> Next step — force-push to GitHub (ONLY if you are happy with"
echo "    the log above):"
echo ""
echo "    git push --force origin main"
echo "================================================================="
