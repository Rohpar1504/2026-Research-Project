#!/usr/bin/env python3
"""
Phase 5e - Graph Encoder Embedding (GEE) + Random Forest classifier (fifth model).

A fifth model family, run ALONGSIDE the GCN (phase5), node2vec + RF (phase5b),
TopER + RF (phase5c), and GGVec + RF (phase5d) - none of which are touched. GEE
(Shen, Wang & Priebe, IEEE TPAMI 2023; actively extended in 2024-2025 follow-up
papers on efficient/sparse and refined GEE) is a fast, deterministic, linear
node embedding - a different mechanism from every other model here: not
random-walk-based (node2vec, GGVec), not topological-filtration-based (TopER),
not a neural net (GCN).

Reference: https://github.com/cshen6/GraphEmd (Experiments/Python/gee/gee.py).
This script calls the unmodified `GraphEncoderEmbedding` class from that package
(installed via `pip install "git+https://github.com/cshen6/GraphEmd.git#subdirectory=Experiments/Python"`);
only the brain-graph loading/CV glue below is project-specific.

How GEE works: unlike every other model here, GEE.fit(X, y) requires a per-node
CLASS LABEL y (not learned - provided). For each node i and class k, its embedding
dimension k = the sum of edge weights from node i to every node in class k,
normalized by class k's size. It is otherwise unsupervised (y is a fixed structural
label, not the AD/CN diagnosis - no leakage).

Label choice for y: each of the 116 AAL regions' HEMISPHERE (Left / Right /
Midline), derived directly and unambiguously from the region name suffix in
data/timeseries/aal_region_labels.csv (`_L`/`_R`, or Vermis_* = midline - verified
116 = 54 L + 54 R + 8 Vermis). This is a real, fixed anatomical property already
present in the data, not invented or derived from the correlation matrices
themselves - and hemispheric connectivity asymmetry is itself an established area
of interest in AD research, making the resulting 3-dim embedding ("how connected is
this region to left-hemisphere / right-hemisphere / midline structures")
scientifically interpretable, not just a technical convenience.

Pipeline position:
    data/connectivity/{method}/{AD,CN}/<subject>.npy   (116 x 116 matrix)
        --> [ graph -> adjacency matrix -> GEE(y=hemisphere) -> mean-pool -> RF ] -->
    leave-one-out accuracy per method  (comparable to phase5b/c/d's numbers)

How it works:
    1. Build a graph from each connectivity matrix (SAME construction as every
       other model: top 20% strongest edges by |correlation|) via the shared
       graph_utils.build_nx_graph, then convert to a dense weighted adjacency
       matrix (nx.to_numpy_array) - GEE takes a matrix directly, no walks needed.
    2. GEE embeds each subject's graph independently (like node2vec/GGVec, unlike
       TopER's joint-cohort computation), using the FIXED hemisphere labels.
    3. Mean-pool the 116 region embeddings into ONE 3-dim feature vector per
       subject (same "global readout" idea as node2vec/GGVec).
    4. Random Forest classifies those vectors, evaluated with leave-one-out CV,
       separately per connectivity method.

Usage:
    .venv/bin/python src/phase5e_gee.py

Requirements: gee (pip install "git+https://github.com/cshen6/GraphEmd.git#subdirectory=Experiments/Python"),
networkx, pandas, scikit-learn, numpy, joblib
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import joblib
from gee.gee import GraphEncoderEmbedding
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import build_nx_graph

# --- Paths & config ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
LABELS_CSV = PROJECT_ROOT / "data" / "timeseries" / "aal_region_labels.csv"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
GROUPS = {"CN": 0, "AD": 1}
METHODS = ["pearson", "spearman", "kendall"]

SEED = 0
N_TREES = 200


def load_hemisphere_labels() -> np.ndarray:
    """
    Derive each of the 116 AAL regions' hemisphere (0=Left, 1=Right, 2=Midline)
    directly from the region name suffix - a real, fixed anatomical property,
    not derived from any subject's connectivity data.
    """
    df = pd.read_csv(LABELS_CSV).sort_values("column")
    def hemi(name: str) -> int:
        if name.endswith("_L"):
            return 0
        elif name.endswith("_R"):
            return 1
        return 2  # Vermis_* (midline)
    return df["region_name"].apply(hemi).to_numpy()


HEMI_LABELS = load_hemisphere_labels()


def embed_subject(mat: np.ndarray) -> np.ndarray:
    """
    Run GEE on one subject's graph (y = fixed hemisphere labels) and return a
    single feature vector for the whole subject (mean of the 116 region
    embeddings, each of dimension 3 = number of hemisphere classes).
    """
    G = build_nx_graph(mat)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n), weight="weight")
    model = GraphEncoderEmbedding()
    # NOTE: model.fit_transform() / model.transform() return X @ pinv(encoder_embedding).T,
    # meant for projecting NEW/unseen nodes onto an already-fit embedding (out-of-sample
    # extension) - NOT the fitted embedding itself. Verified directly: fit_transform's
    # output was a structural constant identical across every subject (a real bug in our
    # usage, caught by checking, not the library). The actual embedding is the
    # `encoder_embedding` attribute set during `.fit()`.
    model.fit(A, HEMI_LABELS[:n])
    emb = model.encoder_embedding
    return emb.mean(axis=0)


def load_features(method: str):
    """Build the GEE feature vector for every subject of one method."""
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
    print("Phase 5e: GEE (hemisphere labels) + Random Forest (leave-one-out CV)")
    print("labels: 0 = CN, 1 = AD\n")

    summary = {}
    for method in METHODS:
        X, y, ids = load_features(method)

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
        joblib.dump(full, MODEL_DIR / f"{method}_gee_rf.joblib")
        print()

    print("Summary - leave-one-out accuracy by method:")
    for method in METHODS:
        print(f"  {method:9s}: {summary[method]:.2f}")


if __name__ == "__main__":
    main()
