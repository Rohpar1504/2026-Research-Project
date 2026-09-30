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
    - note (incl BNT): the Phase 5f Brain Network Transformer is included too, trained
                fresh within each fold like the GCN (both are supervised, end-to-end
                neural nets, unlike the unsupervised-embedding + shallow-classifier
                models above).
    - note 7  : output a table of models x metrics for all three methods.
    - note 2  : report which connectivity method is best for each model.

Metrics: accuracy, sensitivity (AD recall), specificity (CN recall), F1 - each
reported as MEAN +/- STD DEV across the 4 CV folds (computed per-fold, not pooled),
per the advisor's request to include variability alongside the point estimate.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python src/phase6_model_comparison.py

Reuses: node2vec features (phase5b), TopER features (phase5c), GGVec features
(phase5d, via a subprocess bridge - see GGVEC_PYTHON below), GEE features
(phase5e), the GCN model (phase5), and the Brain Network Transformer (phase5f).
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
from phase5f_brain_transformer import (        # the Brain Network Transformer
    BrainNetworkTransformer, BNT_CONFIG,
    train_one as bnt_train_one, predict as bnt_predict,
)

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


def fold_metrics_shallow(model_ctor, X, y, splits) -> list:
    """Per-fold metrics (not pooled) for a scikit-learn model on feature matrix X."""
    results = []
    for train_idx, test_idx in splits:
        clf = model_ctor()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        results.append(compute_metrics(y[test_idx], preds))
    return results


def fold_metrics_gcn(graphs, y, splits) -> list:
    """Per-fold metrics (not pooled) for the GCN, trained fresh within each fold."""
    import torch
    results = []
    in_dim = graphs[0].x.shape[1]
    for train_idx, test_idx in splits:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train_graphs = [graphs[i] for i in train_idx]
        model = train_one(GCN(in_dim), train_graphs)
        preds = np.array([predict(model, graphs[i]) for i in test_idx])
        results.append(compute_metrics(y[test_idx], preds))
    return results


def fold_metrics_bnt(X_raw: np.ndarray, y: np.ndarray, splits) -> list:
    """
    Per-fold metrics (not pooled) for the Brain Network Transformer, trained fresh
    within each fold (same pattern as GCN - both are supervised end-to-end nets).
    X_raw is the RAW (untresholded) connectivity matrices - see phase5f's docstring
    for why BNT uses the full connection profile rather than the thresholded graph.
    """
    import torch
    results = []
    X_t = torch.tensor(X_raw, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)
    for train_idx, test_idx in splits:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = BrainNetworkTransformer(BNT_CONFIG)
        model = bnt_train_one(model, X_t[train_idx], y_t[train_idx])
        preds = np.array([bnt_predict(model, X_t[i]) for i in test_idx])
        results.append(compute_metrics(y[test_idx], preds))
    return results


def aggregate_fold_metrics(fold_results: list) -> dict:
    """
    Mean +/- sample std dev (ddof=1, the standard convention for reporting k-fold CV
    variability) across the 4 per-fold metric dicts, for each metric.
    """
    agg = {}
    for key in fold_results[0]:
        vals = np.array([fr[key] for fr in fold_results])
        agg[key] = vals.mean()
        agg[f"{key}_std"] = vals.std(ddof=1)
    return agg


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
                fold_results = fold_metrics_shallow(ctor, X, y, splits)
                rows.append({"model": f"{feat_name}+{clf_name}", "method": method,
                             **aggregate_fold_metrics(fold_results)})

        # The GCN on graphs.
        fold_results = fold_metrics_gcn(graphs, y, splits)
        rows.append({"model": "GCN", "method": method, **aggregate_fold_metrics(fold_results)})

        # The Brain Network Transformer on the RAW (untresholded) connectivity matrices.
        fold_results = fold_metrics_bnt(np.array(mats), y, splits)
        rows.append({"model": "BrainNetTransformer", "method": method,
                     **aggregate_fold_metrics(fold_results)})

        print(f"  [{method}] done")

    METRIC_COLS = ["accuracy", "accuracy_std", "sensitivity", "sensitivity_std",
                   "specificity", "specificity_std", "f1", "f1_std"]
    df = pd.DataFrame(rows)[["model", "method"] + METRIC_COLS]
    df = df.round(3)

    # Save the full table (abstract numbers - safe to commit).
    out_csv = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(out_csv, index=False)

    print("\n================ MODEL x METHOD COMPARISON (note 7) ================")
    print(df.to_string(index=False))

    # Note 2: best connectivity method for each model (by mean accuracy).
    print("\n---- Best method per model, by accuracy (note 2) ----")
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        best = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {model:24s} -> {best['method']:9s} "
              f"(acc={best['accuracy']:.3f}+/-{best['accuracy_std']:.3f}, "
              f"f1={best['f1']:.3f}+/-{best['f1_std']:.3f})")

    # Note 1: single best model+method overall (by mean accuracy).
    top = df.loc[df["accuracy"].idxmax()]
    print(f"\n---- Best overall (note 1): {top['model']} on {top['method']} "
          f"(acc={top['accuracy']:.3f}+/-{top['accuracy_std']:.3f}, "
          f"f1={top['f1']:.3f}+/-{top['f1_std']:.3f}) ----")

    print(f"\nSaved table -> {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"Reminder: at n={n_subjects} these numbers are indicative, not conclusive.")


if __name__ == "__main__":
    main()
