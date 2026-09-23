#!/usr/bin/env python3
"""
Phase 5b - node2vec + Random Forest classifier (additional model).

A second modeling approach, run ALONGSIDE the GCN (src/phase5_gcn.py is untouched).
Instead of a data-hungry graph neural network, this uses node2vec to extract each
brain graph's structure into a feature vector WITHOUT needing labels, then classifies
AD vs CN with a Random Forest - a structure-aware approach that works at small sample
sizes (where the GCN collapsed).

Pipeline position:
    data/connectivity/{method}/{AD,CN}/<subject>.npy   (116 x 116 matrix)
        -->  [ graph -> node2vec embeddings -> mean-pool -> Random Forest ]  -->
    leave-one-out accuracy per method  (comparable to the GCN's numbers)

How it works:
    1. Build a graph from each connectivity matrix (SAME construction as the GCN:
       top 20% strongest edges by |correlation|, self-loops removed) so both models
       see identical graphs and the comparison is fair.
    2. node2vec takes random walks over the graph and learns an embedding (d numbers)
       for each region capturing its structural role - unsupervised, no labels needed.
    3. Mean-pool the 116 region embeddings into ONE feature vector per subject
       (the same "global readout" idea the GCN used).
    4. Random Forest classifies those vectors, evaluated with leave-one-out CV,
       separately per connectivity method.

Honest caveat:
    node2vec learns each subject's embeddings in its own coordinate space (some random/
    rotational freedom), so embeddings are not perfectly aligned across subjects. Mean-
    pooling + a fixed seed is the standard pragmatic handling; if results look off, we can
    revisit (e.g. joint embedding, or a graph-level method like graph2vec).

Usage:
    .venv/bin/python src/phase5b_node2vec_rf.py

Requirements: node2vec, networkx, scikit-learn, numpy, joblib
"""

import warnings
from pathlib import Path

import sys
import numpy as np
import networkx as nx
import joblib
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import build_nx_graph, KEEP_FRAC  # noqa: F401 (re-exported for phase5c/phase5d)

# --- Paths & config ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
GROUPS = {"CN": 0, "AD": 1}
METHODS = ["pearson", "spearman", "kendall"]

# node2vec settings (standard defaults).
DIM = 64
WALK_LENGTH = 20
NUM_WALKS = 200
SEED = 0

# Random Forest settings.
N_TREES = 200


def embed_subject(mat: np.ndarray) -> np.ndarray:
    """
    Run node2vec on one subject's graph and return a single feature vector for the
    whole subject (mean of the 116 region embeddings).
    """
    G = build_nx_graph(mat)
    n2v = Node2Vec(
        G,
        dimensions=DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        weight_key="weight",   # use edge weights in the random walks
        workers=1,             # single worker => reproducible with the seed
        seed=SEED,
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, seed=SEED)

    # Assemble a (n_regions x DIM) embedding matrix in node order. Isolated nodes
    # (no kept edges) may be absent from the model -> leave them as zeros.
    emb = np.zeros((G.number_of_nodes(), DIM))
    for node in G.nodes():
        key = str(node)
        if key in model.wv:
            emb[node] = model.wv[key]

    return emb.mean(axis=0)     # mean-pool -> one DIM-length vector per subject


def load_features(method: str):
    """Build the node2vec feature vector for every subject of one method."""
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
    print("Phase 5b: node2vec + Random Forest (leave-one-out cross-validation)")
    print("labels: 0 = CN, 1 = AD\n")

    summary = {}
    for method in METHODS:
        # node2vec embeddings are unsupervised, so compute each subject's feature
        # vector ONCE (independent of the train/test split), then do LOOCV on top.
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

        # Save a Random Forest trained on all subjects (reference artifact).
        full = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED).fit(X, y)
        joblib.dump(full, MODEL_DIR / f"{method}_node2vec_rf.joblib")
        print()

    print("Summary - leave-one-out accuracy by method:")
    for method in METHODS:
        print(f"  {method:9s}: {summary[method]:.2f}")


if __name__ == "__main__":
    main()
