#!/usr/bin/env python3
"""
Bridge script for Phase 6: computes GGVec features for one connectivity method and
writes them to a .npz file, so phase6_model_comparison.py (running in the MAIN venv,
which cannot import nodevectors - see requirements-nodevectors.txt / README) can pull
them in as a subprocess call into the SEPARATE ~/.venvs/nodevectors environment.

This is plumbing only: it just calls phase5d_nodevectors.py's own (unmodified)
load_features() and serializes the result. No embedding logic lives here.

Usage (invoked automatically by phase6_model_comparison.py; not meant to be run by
hand, but harmless to if you want to inspect it directly):
    ~/.venvs/nodevectors/bin/python src/ggvec_bridge.py --method pearson --out /tmp/x.npz
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5d_nodevectors import load_features


def main() -> None:
    warnings.filterwarnings("ignore")  # GGVec's own convergence warnings are just noise here
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["pearson", "spearman", "kendall"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    X, y, ids = load_features(args.method)
    np.savez(args.out, X=X, y=y, ids=np.array(ids))


if __name__ == "__main__":
    main()
