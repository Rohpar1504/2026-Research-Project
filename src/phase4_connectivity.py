#!/usr/bin/env python3
"""
Phase 4 - Connectivity matrix construction (the CORE of the project).

For each subject, converts the (timepoints x regions) time-series from Phase 3 into
THREE (regions x regions) functional-connectivity matrices - one per correlation
method. Entry (i, j) of a matrix = the correlation between region i's and region j's
activity over time. These matrices are the "edges" of each subject's brain network,
and the correlation method is the single variable this project tests.

Pipeline position:
    data/timeseries/{AD,CN}/<subject>.npy         (matrix: timepoints x regions)
        -->  [ Pearson / Spearman / Kendall ]  -->
    data/connectivity/{method}/{AD,CN}/<subject>.npy   (matrix: regions x regions)

The three methods:
    Pearson  - linear correlation (do signals rise/fall together proportionally?).
               Computed via nilearn ConnectivityMeasure, as specified by the advisor.
               NOTE: we pass an EmpiricalCovariance estimator so this is a *pure*
               Pearson correlation. nilearn's default (LedoitWolf) applies shrinkage,
               which would regularize Pearson but not Spearman/Kendall - an unfair
               advantage that would confound the comparison. Empirical = no shrinkage.
    Spearman - rank correlation (monotonic relationship); robust to outliers. scipy.
    Kendall  - rank correlation from concordant/discordant pairs; most robust,
               conservative, and computed pairwise (no vectorized version). scipy.

Design choices (from the Phase 4 proposal):
    - Diagonal kept as 1 (raw correlation); Phase 5 decides self-loops.
    - Signed correlations kept (negative = anti-correlation); no absolute value.

Usage (project venv has scipy + nilearn):
    .venv/bin/python src/phase4_connectivity.py
"""

import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt

from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.covariance import EmpiricalCovariance
from nilearn.connectome import ConnectivityMeasure

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TS_DIR = PROJECT_ROOT / "data" / "timeseries"          # input:  time-series
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"      # output: connectivity matrices
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "connectivity"
GROUPS = ["AD", "CN"]
METHODS = ["pearson", "spearman", "kendall"]


# --- One function per correlation method (each takes (timepoints x regions),
#     returns a (regions x regions) matrix) ------------------------------------

def pearson_matrix(ts: np.ndarray) -> np.ndarray:
    """Pure Pearson correlation via nilearn, using empirical (unshrunk) covariance."""
    measure = ConnectivityMeasure(
        kind="correlation",
        cov_estimator=EmpiricalCovariance(store_precision=False),
        standardize=False,   # signals already z-scored in Phase 3; correlation is
                             # scale-invariant anyway, so this changes nothing but
                             # avoids double-standardizing.
    )
    # fit_transform expects a list of subjects; we pass one and take element [0].
    return measure.fit_transform([ts])[0]


def spearman_matrix(ts: np.ndarray) -> np.ndarray:
    """Spearman rank correlation. One scipy call returns the full matrix."""
    rho, _ = spearmanr(ts)          # columns treated as variables -> (regions, regions)
    return np.asarray(rho)


def kendall_matrix(ts: np.ndarray) -> np.ndarray:
    """
    Kendall's tau. No vectorized matrix version exists, so compute each unique
    region pair once and mirror it (the matrix is symmetric). Diagonal = 1.
    """
    n = ts.shape[1]
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            tau, _ = kendalltau(ts[:, i], ts[:, j])
            mat[i, j] = mat[j, i] = tau
    return mat


METHOD_FUNCS = {
    "pearson": pearson_matrix,
    "spearman": spearman_matrix,
    "kendall": kendall_matrix,
}


def verify(mat: np.ndarray, n_regions: int) -> dict:
    """Sanity checks a connectivity matrix should pass. Returns {check: bool}."""
    return {
        "shape": mat.shape == (n_regions, n_regions),
        "symmetric": np.allclose(mat, mat.T, atol=1e-6),
        "diag~1": np.allclose(np.diag(mat), 1.0, atol=1e-6),
        "in[-1,1]": np.nanmin(mat) >= -1.001 and np.nanmax(mat) <= 1.001,
        "no_nan": not np.isnan(mat).any(),
    }


def save_heatmaps(mats: dict, subject_id: str, group: str, out_path: Path) -> None:
    """Plot the three methods' matrices side by side for one subject."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    im = None
    for ax, method in zip(axes, METHODS):
        im = ax.imshow(mats[method], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(method.capitalize())
        ax.set_xlabel("region"); ax.set_ylabel("region")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="correlation")
    fig.suptitle(f"{group}: {subject_id} - functional connectivity by method")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Phase 4: connectivity matrix construction\n")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    total, all_ok = 0, True
    for group in GROUPS:
        ts_files = sorted((TS_DIR / group).glob("*.npy"))
        print(f"[{group}] {len(ts_files)} subject(s)")

        for ts_file in ts_files:
            total += 1
            subject_id = ts_file.stem
            ts = np.load(ts_file)                 # (timepoints, regions)
            n_regions = ts.shape[1]

            mats = {}
            for method in METHODS:
                mat = METHOD_FUNCS[method](ts)
                mats[method] = mat

                # Save the matrix under data/connectivity/{method}/{group}/
                out_dir = CONN_DIR / method / group
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"{subject_id}.npy", mat)

                # Verify and report any failing checks.
                checks = verify(mat, n_regions)
                failed = [k for k, ok in checks.items() if not ok]
                if failed:
                    all_ok = False
                    print(f"    {subject_id} [{method}]: FAILED {failed}")

            # One comparison figure per subject.
            save_heatmaps(mats, subject_id, group, FIG_DIR / f"{group}_{subject_id}.png")
            print(f"    {subject_id}: 3 matrices ({n_regions}x{n_regions}) saved + heatmap")
        print()

    print(f"Done: {total} subjects x 3 methods = {total * 3} connectivity matrices.")
    if not all_ok:
        sys.exit("ERROR: some matrices failed verification (see above).")


if __name__ == "__main__":
    main()
