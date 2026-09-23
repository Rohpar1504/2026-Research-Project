#!/usr/bin/env python3
"""
Phase 6 - Model & method comparison (research-meeting action items 2, 3, 7, 11).

Builds the main comparison TABLE: how well does each model classify AD vs CN, across
each of the three connectivity methods (Pearson / Spearman / Kendall)?

Addresses:
    - note 3  : 4-fold stratified cross-validation (k=4), suited to the small cohort,
                replacing leave-one-out. The SAME folds are used for every model.
    - note 11 : compare classifiers - Random Forest vs MLP vs Logistic Regression
                (the "sigmoid shallow classifier") - on node2vec, TopER (Phase 5c),
                GGVec/nodevectors (Phase 5d), and GEE (Phase 5e) features.
    - note (incl GCN): the Phase 5 Graph Convolutional Network is included too, run on
                the graphs directly under the same folds.
    - note 7  : output a table of models x metrics for all three methods.
    - note 2  : report which connectivity method is best for each model.

Metrics: accuracy, sensitivity (AD recall), specificity (CN recall), F1.
All metrics are computed from OUT-OF-FOLD predictions (every subject predicted once,
while held out), pooled over the 4 folds - stabler than averaging tiny per-fold scores.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python src/phase6_model_comparison.py

Reuses: node2vec features (phase5b), TopER features (phase5c), GGVec features
(phase5d, via a subprocess bridge - see GGVEC_PYTHON below), GEE features
(phase5e), and the GCN model (phase5).
"""

import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Make the sibling phase scripts importable (they live in the same src/ folder).
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from phase5b_node2vec_rf import embed_subject          # node2vec -> per-subject vector
from phase5c_toper import embed_method as embed_toper_method  # TopER -> per-subject vector
from phase5e_gee import embed_subject as embed_gee_subject  # GEE -> per-subject vector
from phase5_gcn import GCN, build_graph, train_one, predict  # the graph model

# --- Paths & config ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
GROUPS = {"CN": 0, "AD": 1}
METHODS = ["pearson", "spearman", "kendall"]

# GGVec (nodevectors) lives in a SEPARATE venv (networkx version conflict - see
# README "Note on Phase 5d"), so its features are computed via subprocess into
# src/ggvec_bridge.py rather than imported directly into this process.
GGVEC_PYTHON = Path.home() / ".venvs" / "nodevectors" / "bin" / "python"

N_SPLITS = 4     # 4-fold cross-validation (note 3)
SEED = 0

# The shallow classifiers (note 11). Each is a fresh instance per fold. Applied
# identically to every feature source below, so the comparison stays fair.
CLASSIFIER_CTORS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=SEED),
    "MLP":          lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=SEED),
    "LogisticReg":  lambda: LogisticRegression(max_iter=1000, random_state=SEED),
}


def load_method(method: str):
    """Load all subjects' connectivity matrices + labels + ids for one method."""
    mats, y, ids = [], [], []
    for group, label in GROUPS.items():
        for f in sorted((CONN_DIR / method / group).glob("*.npy")):
            mats.append(np.load(f))
            y.append(label)
            ids.append(f"{group}/{f.stem}")
    return mats, np.array(y), ids


def embed_ggvec_method(method: str, expected_ids: list):
    """
    GGVec features for one method, computed by shelling out to the SEPARATE
    ~/.venvs/nodevectors environment (src/ggvec_bridge.py), since nodevectors'
    networkx==2.5.1 pin cannot coexist in this process with node2vec's/toper's
    networkx>=3. Returns X only; verifies subject order matches load_method's.
    """
    if not GGVEC_PYTHON.exists():
        raise FileNotFoundError(
            f"GGVec venv not found at {GGVEC_PYTHON}. Set it up per the README "
            f"'Note on Phase 5d' (python3 -m venv ~/.venvs/nodevectors ; "
            f"~/.venvs/nodevectors/bin/pip install -r requirements-nodevectors.txt)."
        )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"{method}.npz"
        subprocess.run(
            [str(GGVEC_PYTHON), str(SRC_DIR / "ggvec_bridge.py"),
             "--method", method, "--out", str(out_path)],
            check=True,
        )
        data = np.load(out_path, allow_pickle=True)
        X, ids = data["X"], list(data["ids"])
    assert ids == expected_ids, "GGVec subject order must match load_method's order"
    return X


def compute_metrics(y_true, y_pred) -> dict:
    """accuracy, sensitivity (AD=1 recall), specificity (CN=0 recall), F1 (for AD)."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1":          f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def oof_predictions_shallow(model_ctor, X, y, splits) -> np.ndarray:
    """Out-of-fold predictions for a scikit-learn model on feature matrix X."""
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in splits:
        clf = model_ctor()
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    return preds


def oof_predictions_gcn(graphs, y, splits) -> np.ndarray:
    """Out-of-fold predictions for the GCN, trained on graphs under the same folds."""
    import torch
    preds = np.zeros(len(y), dtype=int)
    in_dim = graphs[0].x.shape[1]
    for train_idx, test_idx in splits:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train_graphs = [graphs[i] for i in train_idx]
        model = train_one(GCN(in_dim), train_graphs)
        for i in test_idx:
            preds[i] = predict(model, graphs[i])
    return preds


def main() -> None:
    warnings.filterwarnings("ignore")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 6: model & method comparison (4-fold stratified CV)\n")

    rows = []
    n_subjects = 0
    for method in METHODS:
        mats, y, ids = load_method(method)
        n_subjects = len(y)

        # Five representations of the same subjects:
        #   - node2vec feature vectors (for the shallow models) - computed once
        #   - TopER feature vectors (for the shallow models) - computed once
        #   - GGVec feature vectors (for the shallow models) - computed once, via
        #     subprocess into the separate nodevectors venv
        #   - GEE feature vectors (for the shallow models) - computed once
        #   - PyG graphs (for the GCN)
        X_node2vec = np.array([embed_subject(m) for m in mats])
        X_toper, y_toper, ids_toper = embed_toper_method(method)
        assert ids_toper == ids, "TopER subject order must match load_method's order"
        X_ggvec = embed_ggvec_method(method, ids)
        X_gee = np.array([embed_gee_subject(m) for m in mats])
        graphs = [build_graph(m, int(lbl)) for m, lbl in zip(mats, y)]

        # One fixed set of 4 stratified folds, reused by EVERY model for fairness.
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(skf.split(np.zeros(len(y)), y))

        # Shallow models on node2vec, TopER, GGVec, and GEE features in turn.
        for feat_name, X in [("node2vec", X_node2vec), ("TopER", X_toper),
                              ("GGVec", X_ggvec), ("GEE", X_gee)]:
            for clf_name, ctor in CLASSIFIER_CTORS.items():
                preds = oof_predictions_shallow(ctor, X, y, splits)
                rows.append({"model": f"{feat_name}+{clf_name}", "method": method,
                             **compute_metrics(y, preds)})

        # The GCN on graphs.
        preds = oof_predictions_gcn(graphs, y, splits)
        rows.append({"model": "GCN", "method": method, **compute_metrics(y, preds)})

        print(f"  [{method}] done")

    df = pd.DataFrame(rows)[["model", "method", "accuracy", "sensitivity", "specificity", "f1"]]
    df = df.round(3)

    # Save the full table (abstract numbers - safe to commit).
    out_csv = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(out_csv, index=False)

    print("\n================ MODEL x METHOD COMPARISON (note 7) ================")
    print(df.to_string(index=False))

    # Note 2: best connectivity method for each model (by accuracy).
    print("\n---- Best method per model, by accuracy (note 2) ----")
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        best = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {model:24s} -> {best['method']:9s} (acc={best['accuracy']:.3f}, f1={best['f1']:.3f})")

    # Note 1: single best model+method overall (by accuracy).
    top = df.loc[df["accuracy"].idxmax()]
    print(f"\n---- Best overall (note 1): {top['model']} on {top['method']} "
          f"(acc={top['accuracy']:.3f}, f1={top['f1']:.3f}) ----")

    print(f"\nSaved table -> {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"Reminder: at n={n_subjects} these numbers are indicative, not conclusive.")


if __name__ == "__main__":
    main()
