#!/usr/bin/env python3
"""
Phase 5d - GGVec (nodevectors) + Random Forest classifier (fourth modeling approach).

A fourth model family, run ALONGSIDE the GCN (phase5), node2vec + RF (phase5b), and
TopER + RF (phase5c) - none of which are touched. GGVec is a node-embedding algorithm
from the `nodevectors` package (https://github.com/VHRanger/nodevectors): unlike
node2vec's random-walk + skip-gram approach, GGVec directly minimizes a loss over edge
weights via alternating relaxation (negative sampling) and contraction passes - closer
in spirit to GloVe than to word2vec. A genuinely different mechanism from node2vec, not
a re-implementation of it.

Reference: https://github.com/VHRanger/nodevectors. This script calls the unmodified
`nodevectors.GGVec` from the pip package (`pip install nodevectors`); only the
brain-graph loading/CV glue below is project-specific.

REQUIRES A SEPARATE VIRTUAL ENVIRONMENT from the rest of the project. nodevectors'
own dependency (csrgraph) calls the removed nx.adj_matrix() function, so it hard-pins
networkx==2.5.1 - incompatible with node2vec's and toper's networkx>=3 requirement.
That old networkx in turn needs numpy<2.0, which in turn needs scipy<1.14. There is no
way to satisfy all four packages' pins in one environment, so this script imports
ONLY from graph_utils.py (dependency-free) rather than phase5b_node2vec_rf.py (which
would drag in gensim/node2vec and fail to import here).

Pipeline position:
    data/connectivity/{method}/{AD,CN}/<subject>.npy   (116 x 116 matrix)
        --> [ graph -> GGVec embeddings -> mean-pool -> Random Forest ] -->
    leave-one-out accuracy per method  (comparable to phase5b/phase5c's numbers)

How it works:
    1. Build a graph from each connectivity matrix (SAME construction as every other
       model: top 20% strongest edges by |correlation|, self-loops removed) via the
       shared graph_utils.build_nx_graph - identical graphs across all four models.
    2. GGVec embeds each subject's graph independently (like node2vec, unlike TopER's
       joint-cohort computation) - one call per subject.
    3. Mean-pool the 116 region embeddings into ONE feature vector per subject (same
       "global readout" idea node2vec uses).
    4. Random Forest classifies those vectors, evaluated with leave-one-out CV,
       separately per connectivity method.

Usage (MUST use the dedicated venv, not the project's main .venv):
    ~/.venvs/nodevectors/bin/python src/phase5d_nodevectors.py

Requirements (in ~/.venvs/nodevectors only): nodevectors, networkx==2.5.1,
numpy<2.0, scipy<1.14, scikit-learn, joblib
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import joblib
from nodevectors import GGVec
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import build_nx_graph

# --- Paths & config ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
GROUPS = {"CN": 0, "AD": 1}
METHODS = ["pearson", "spearman", "kendall"]

# GGVec settings - library defaults (nodevectors/ggvec.py), not re-tuned.
DIM = 32
SEED = 0

# Random Forest settings - same as phase5b/phase5c, for a fair comparison.
N_TREES = 200


def embed_subject(mat: np.ndarray) -> np.ndarray:
    """
    Run GGVec on one subject's graph and return a single feature vector for the
    whole subject (mean of the 116 region embeddings).
    """
    G = build_nx_graph(mat)
    model = GGVec(n_components=DIM, verbose=False)
    vectors = model.fit_transform(G)   # (n_nodes, DIM), in G.nodes() order
    return vectors.mean(axis=0)


def load_features(method: str):
    """Build the GGVec feature vector for every subject of one method."""
    X, y, ids = [], [], []
    for group, label in GROUPS.items():
        for f in sorted((CONN_DIR / method / group).glob("*.npy")):
            X.append(embed_subject(np.load(f)))
            y.append(label)
            ids.append(f"{group}/{f.stem}")
    return np.array(X), np.array(y), ids


def main() -> None:
    warnings.filterwarnings("ignore")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 5d: GGVec (nodevectors) + Random Forest (leave-one-out CV)")
    print("labels: 0 = CN, 1 = AD\n")

    summary = {}
    for method in METHODS:
        X, y, ids = load_features(method)

        nan_rows = np.isnan(X).any(axis=1)
        if nan_rows.any():
            print(f"    [ggvec] {nan_rows.sum()} subject(s) produced NaN for "
                  f"{method}: {[ids[i] for i in np.where(nan_rows)[0]]}")

        preds = []
        for i in range(len(X)):
            train_idx = [j for j in range(len(X)) if j != i]
            clf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED)
            clf.fit(X[train_idx], y[train_idx])
            preds.append(int(clf.predict(X[i:i + 1])[0]))

        correct = sum(p == t for p, t in zip(preds, y))
        acc = correct / len(y)
        summary[method] = acc
        print(f"[{method}] leave-one-out accuracy = {acc:.2f}  ({correct}/{len(y)})")
        for sid, p, t in zip(ids, preds, y):
            print(f"    {sid:16s} true={t} pred={p}  {'ok' if p == t else 'MISS'}")

        full = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED).fit(X, y)
        joblib.dump(full, MODEL_DIR / f"{method}_ggvec_rf.joblib")
        print()

    print("Summary - leave-one-out accuracy by method:")
    for method in METHODS:
        print(f"  {method:9s}: {summary[method]:.2f}")


if __name__ == "__main__":
    main()
