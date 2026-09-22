#!/usr/bin/env python3
"""
Phase 5c - TopER + Random Forest classifier (third modeling approach).

A third model family, run ALONGSIDE the GCN (src/phase5_gcn.py) and node2vec + RF
(src/phase5b_node2vec_rf.py) - both untouched. TopER (Topological Evolution Rate)
is a graph-level, topology-inspired embedding: for each filtration function it fits
a line to (node/edge count vs threshold) and returns just the line's [intercept,
slope] - a handful of interpretable numbers per subject, instead of node2vec's
64-dim mean-pooled vector.

Reference: https://github.com/AstritTola/TopER (NeurIPS 2025). This script calls
the unmodified `toper.main()` from the pip package (`pip install toper`); only the
brain-graph loading/CV glue below is project-specific.

Pipeline position:
    data/connectivity/{method}/{AD,CN}/<subject>.npy   (116 x 116 matrix)
        --> [ graphs -> toper.main() -> per-subject feature vector -> Random Forest ] -->
    leave-one-out accuracy per method  (comparable to phase5b's numbers)

How it works:
    1. Build a graph from each connectivity matrix (SAME construction as the GCN and
       node2vec: top 20% strongest edges by |correlation|, self-loops removed, edge
       weight = |correlation|) so every model in the project sees identical graphs.
    2. Unlike node2vec (embedded one subject at a time), TopER computes its
       thresholds JOINTLY across every graph passed into one call - so all subjects
       for a method are embedded in a single `toper.main()` call, not a loop.
    3. For each filtration function, TopER returns a [a, b] pair per subject (both
       "sub-level" and "super-level" filtrations). We concatenate the pairs across
       functions/levels into one feature vector per subject - no pooling needed,
       since TopER is already graph-level.
    4. Random Forest classifies those vectors, evaluated with leave-one-out CV,
       separately per connectivity method.

Honest caveat:
    TopER's thresholds are shared across whatever cohort is passed into one call, so
    (unlike node2vec) a subject's feature vector can shift slightly if the cohort
    changes - still unsupervised (no labels involved), so no leakage into the CV
    split, but worth noting alongside node2vec's own alignment caveat in the README.

    Separately: toper/utils.py's best_fit_u1mean0/best_fit_u0mean1 (the closed-form
    least-squares fit TopER uses internally) divides by `A*C - B**2`, which is exactly
    zero for a subject whose thresholded graph produces a degenerate (zero-variance)
    count sequence - a real numerical edge case in the library, not a bug in this
    script. Observed for 3/28 subjects (consistently across every filtration/level) on
    the pearson connectivity method. We impute those NaN rows with the per-column
    median (sklearn's SimpleImputer) rather than silently letting a downstream
    classifier tolerate or reject them.

Usage:
    .venv/bin/python src/phase5c_toper.py

Requirements: toper, GraphRicciCurvature (toper's own dependency), networkx,
scikit-learn, numpy, joblib
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import joblib
import toper
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Reuse the SAME graph construction as node2vec/GCN, unmodified.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5b_node2vec_rf import build_nx_graph

# --- Paths & config ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
GROUPS = {"CN": 0, "AD": 1}
METHODS = ["pearson", "spearman", "kendall"]

# TopER settings.
# deg_cen: node degree centrality (pure topology, cheap baseline).
# fricci:  Forman-Ricci edge curvature - reads the "weight" edge attribute, which
#          build_nx_graph already sets to |correlation|, so this filtration is the
#          one that actually uses connection STRENGTH, not just thresholded topology.
FUNCT_LIST = ["deg_cen", "fricci"]
# NOTE: toper==1.0.3's reduce_thresholds() (toper/threshold_reduction.py) calls
# reduce_list(thresholds) without its required num_segm argument, which raises
# TypeError whenever a filtration's raw threshold count exceeds num_segm (this
# happens for fricci's curvature values, though not for deg_cen). This is a bug in
# the published library, not our code - the workaround is to keep num_segm above
# the raw threshold count so reduce_thresholds' short-circuit ("len <= max_len")
# always returns early and the buggy branch is never reached. It does not change
# TopER's methodology (embedding dimensionality is fixed at 2 numbers per
# function/level regardless of num_segm) - it only raises the threshold resolution.
NUM_SEGM = 100_000

# Random Forest settings - same as phase5b, for a fair comparison.
SEED = 0
N_TREES = 200


def build_graph_list(method: str):
    """
    Build the graph list (+ labels/ids) for every subject of one method, in a fixed
    order. TopER needs the WHOLE list at once (unlike node2vec's per-subject loop),
    since it computes its thresholds jointly across every graph passed in.
    """
    graphs, y, ids = [], [], []
    for group, label in GROUPS.items():
        for f in sorted((CONN_DIR / method / group).glob("*.npy")):
            graphs.append(build_nx_graph(np.load(f)))
            y.append(label)
            ids.append(f"{group}/{f.stem}")
    return graphs, np.array(y), ids


# toper.main()'s returned dict keys don't always match the funct_list names passed
# in - toper/main.py's run_forricci()/run_olricci() hardcode "forricci"/"olricci"
# as the output prefix even though the funct_list entries are "fricci"/"oricci".
# Confirmed by inspecting the actual KeyError at runtime, not assumed.
OUTPUT_KEY_PREFIX = {
    "degree": "degree",
    "deg_cen": "deg_cen",
    "popularity": "popularity",
    "closeness": "closeness",
    "fricci": "forricci",
    "oricci": "olricci",
    "weight": "weight",
}


def embed_method(method: str):
    """
    Run TopER once on every subject's graph for one method, and assemble each
    subject's [a, b] pairs (per function, per sub/super level) into one feature
    vector. With FUNCT_LIST = ["deg_cen", "fricci"] that's 2 functions x 2 levels
    x 2 numbers = 8 features per subject.
    """
    graphs, y, ids = build_graph_list(method)
    F = toper.main(graphs, FUNCT_LIST, num_segm=NUM_SEGM)

    X = np.array([
        [val for func in FUNCT_LIST
             for level in ("sub", "super")
             for val in F[f"{OUTPUT_KEY_PREFIX[func]}_{level}"][i]]
        for i in range(len(graphs))
    ])

    # See "Honest caveat" above: toper's own least-squares fit divides by zero for
    # subjects whose thresholded graph is degenerate, producing all-NaN rows. Report
    # and impute (median per column) rather than let it pass through silently.
    nan_rows = np.isnan(X).any(axis=1)
    if nan_rows.any():
        print(f"    [toper] {nan_rows.sum()} subject(s) hit a degenerate TopER fit "
              f"(NaN) for {method}: {[ids[i] for i in np.where(nan_rows)[0]]} "
              f"- imputing with per-column median.")
        X = SimpleImputer(strategy="median").fit_transform(X)

    return X, y, ids


def main() -> None:
    warnings.filterwarnings("ignore")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 5c: TopER + Random Forest (leave-one-out cross-validation)")
    print("labels: 0 = CN, 1 = AD\n")

    summary = {}
    for method in METHODS:
        # TopER embeddings are unsupervised, so compute every subject's feature
        # vector ONCE per method (independent of the train/test split), then do
        # LOOCV on top - same pattern as phase5b's node2vec features.
        X, y, ids = embed_method(method)

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
        joblib.dump(full, MODEL_DIR / f"{method}_toper_rf.joblib")
        print()

    print("Summary - leave-one-out accuracy by method:")
    for method in METHODS:
        print(f"  {method:9s}: {summary[method]:.2f}")


if __name__ == "__main__":
    main()
