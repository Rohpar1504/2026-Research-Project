# Does the Correlation Method Matter? Pearson vs. Spearman vs. Kendall Functional Connectivity for Alzheimer's Classification

**Author:** Rohan Pareek — Graduate Researcher, Siebel School of Computing and Data Science, UIUC<br>
**Advisor:** Prof. Pablo D. Robles-Granda

## Research Question
Does the choice of correlation method (Pearson vs. Spearman vs. Kendall) for building
brain functional-connectivity networks from fMRI meaningfully affect downstream
classification of Alzheimer's Disease (AD) vs. Cognitively Normal (CN) subjects?
The comparison is evaluated across several graph-based classification models.

## Dataset
ADNI (Alzheimer's Disease Neuroimaging Initiative) — **28 subjects, balanced 14 AD / 14 CN**,
all **ADNI-2 resting-state fMRI** (3T, TR ≈ 3 s, "Resting State fMRI" protocol).
Groups are **sex-mirrored** (each 8M/6F) and **age-matched** (AD ≈ 78.3, CN ≈ 78.6).
**No MCI or any other condition** — a clean binary split. Data begins as raw DICOM.

## Pipeline
| Phase | What it does | Script | Status |
|-------|--------------|--------|--------|
| 1 | Setup, data acquisition & organization | — (scaffolding) | ✅ |
| 2 | DICOM → NIfTI conversion (`dcm2niix`) | `src/phase2_dicom_to_nifti.py` | ✅ |
| 3 | AAL atlas parcellation → region time-series (`NiftiLabelsMasker`) | `src/phase3_extract_timeseries.py` | ✅ |
| — | QC: visualize atlas-on-brain alignment | `src/qc_atlas_overlay.py` | ✅ |
| 4 | Build 3 connectivity matrices/subject (Pearson/Spearman/Kendall) | `src/phase4_connectivity.py` | ✅ |
| 5 | Graph construction + **GCN** classifier | `src/phase5_gcn.py` | ✅ |
| 5b | **node2vec** embeddings + Random Forest classifier | `src/phase5b_node2vec_rf.py` | ✅ |
| 5c | **TopER** (topology-inspired graph embedding) + Random Forest classifier | `src/phase5c_toper.py` | ✅ |
| 5d | **GGVec** (nodevectors) embeddings + Random Forest classifier | `src/phase5d_nodevectors.py` | ✅ |
| 5e | **GEE** (Graph Encoder Embedding) + Random Forest classifier | `src/phase5e_gee.py` | ✅ |
| 6 | 4-fold CV comparison: models × methods × metrics | `src/phase6_model_comparison.py` | ✅ |
| 7 | Interpret important regions/edges per method | (planned) | ⏭️ |
| 8 | Write-up | `outputs/` | ⏭️ |

Each script is **independently runnable** and reads from the previous phase's output folder.
Run scripts with the project venv, e.g. `.venv/bin/python src/phase4_connectivity.py`.

## Models Compared
Five families of graph-based models classify each subject's connectivity network:

- **GCN** (`phase5_gcn.py`) — a Graph Convolutional Network + MLP head that learns
  directly from the graph structure. *Data-hungry; collapses to predicting one class
  (sensitivity 0.0) even at n=28.*
- **node2vec + classifier** (`phase5b`, `phase6`) — node2vec embeds each graph's
  structure (unsupervised), mean-pooled into a per-subject vector, then classified by
  **Random Forest**, **MLP**, or **Logistic Regression** (a sigmoid shallow classifier).
- **TopER + classifier** (`phase5c`, `phase6`) — TopER (Topological Evolution Rate;
  Tola et al., NeurIPS 2025) is a graph-*level* topological embedding: for each
  filtration function (`deg_cen` degree centrality, `fricci` Forman-Ricci curvature —
  the latter uses edge weight = `|correlation|`) it fits a line across thresholds and
  returns just `[intercept, slope]` per sub/super-level, giving an 8-dimensional,
  low-dimensional, interpretable feature vector per subject — no pooling needed, unlike
  node2vec. Classified the same way: **Random Forest**, **MLP**, or **Logistic
  Regression**.
- **GGVec + classifier** (`phase5d`, `phase6` via `ggvec_bridge.py`) — GGVec, from the
  `nodevectors` package (Ranger), embeds each node by directly minimizing a loss over
  edge weights via alternating relaxation/contraction passes — no random walks,
  closer in spirit to GloVe than to node2vec's skip-gram approach. Mean-pooled into a
  per-subject vector like node2vec, classified by **Random Forest**, **MLP**, or
  **Logistic Regression**. **Runs in a separate virtual environment**
  (`~/.venvs/nodevectors` — see Setup) because its `csrgraph` dependency hard-pins
  `networkx==2.5.1`, incompatible with node2vec's and TopER's `networkx>=3`
  requirement — `phase6` bridges it in via a subprocess call to `ggvec_bridge.py`
  rather than importing it directly, so it still appears in the unified comparison
  table. `phase5d_nodevectors.py` also runs it standalone (leave-one-out CV) if you
  just want GGVec's numbers on their own.
- **GEE + classifier** (`phase5e`, `phase6`) — Graph Encoder Embedding (Shen, Wang &
  Priebe, IEEE TPAMI 2023; actively extended in 2024-2025 follow-up papers) is a
  fast, deterministic, linear embedding - a mechanism different from every other
  model here (not walk-based, not filtration-based, not a neural net). Unlike the
  other unsupervised embeddings, GEE requires a per-node CLASS LABEL as input: we
  use each AAL region's **hemisphere** (Left/Right/Midline), derived directly and
  unambiguously from the region name suffix in `aal_region_labels.csv` - a real,
  fixed anatomical property already in the data (54 L / 54 R / 8 Vermis-midline),
  not invented or derived from the correlation matrices. Each node's embedding
  dimension *k* = its total edge weight to every node in hemisphere *k*, giving a
  3-dimensional, directly interpretable feature per region ("how connected is this
  region to left/right/midline structures") - mean-pooled into a per-subject vector
  and classified the same way as the others.

All models see identical graphs (top-20% strongest edges by `|correlation|`) so any
performance difference reflects the model or the connectivity method, not preprocessing.

## Current Status & Findings (indicative, n = 28)
Phase 6 evaluates every model × method — **13 models × 3 methods = 39 rows** (GCN,
node2vec, TopER, GGVec, and GEE) — with **4-fold stratified cross-validation**,
reporting accuracy, sensitivity, specificity, and F1
(`outputs/results/model_comparison.csv`).

- **Best result overall is node2vec + MLP on Spearman** — accuracy **0.857**, F1
  **0.857**. This is the most credible finding: it has held up (and slightly improved)
  across every rerun, including after adding TopER, fixing a graph-construction bug
  (below), and adding GGVec and GEE.
- **GGVec's best result under 4-fold CV is GGVec + RandomForest on Kendall** —
  accuracy **0.571**. Note `phase5d_nodevectors.py`'s own **leave-one-out**
  evaluation showed GGVec + Random Forest reaching **0.79** on Pearson in an earlier
  run - that gap is partly the LOOCV-vs-4-fold-CV protocol difference (worth
  discussing with the advisor), and partly because **GGVec's own class has no fixed
  random seed** (confirmed from its source - `nodevectors.GGVec` takes no
  `random_state`/`seed` parameter), so its embeddings - and downstream accuracy -
  vary somewhat between runs even with identical code and data. Treat any single
  GGVec number as approximate; a multi-run average would be more honest than any one
  run's figure.
- **TopER's best results tie at accuracy 0.50** (TopER + MLP on Spearman and on
  Kendall; TopER + LogisticReg on Pearson) — meaningfully weaker than node2vec here.
  TopER does **not** reproduce node2vec's Spearman signal, reinforcing that the
  "Spearman helps" effect is tied to specific representations, not a
  method-independent property of the data.
- **GEE's best result is GEE + RandomForest on Kendall** — accuracy **0.536**, F1
  **0.435**. GEE is deterministic (no randomness in its embedding step, unlike
  GGVec), so this number is fully reproducible. Its 3-dimensional hemisphere-based
  embedding is the lowest-dimensional and most directly interpretable of any model
  here, but also the weakest signal overall - a reasonable outcome given hemisphere
  alone is a coarse structural summary compared to the other methods' richer
  representations.
- **The GCN fully collapses** — sensitivity **0.000** across all three methods (predicts
  every subject CN) — unaffected by the fix below, since it uses its own separate graph
  builder in `phase5_gcn.py`. Too data-hungry at this scale, unchanged conclusion.
- **Data-quality bug found and fixed (2026):** `graph_utils.build_nx_graph` (the graph
  construction shared by node2vec/TopER/GGVec/GEE) computed its edge threshold with
  `np.quantile`, which silently returns `NaN` when its input contains `NaN` values.
  Three subjects (`CN/010_S_4442`, `AD/013_S_5071`, `AD/018_S_4733`) have `NaN` entries
  in their connectivity matrices themselves (undefined correlation for some region
  pair, likely a near-constant region time-series from Phase 3) across **all three**
  correlation methods — every edge then silently failed the (NaN) threshold check,
  producing a completely edgeless graph for those subjects. node2vec and TopER
  degraded silently (this is what caused TopER's earlier-reported NaN-fit issue);
  GGVec's dependency crashed outright on the empty edge list, which is how this was
  actually caught. Fixed with `np.nanquantile` + explicitly skipping NaN-valued pairs
  (see `graph_utils.py`); all `phase6` numbers in this README reflect the fix. Worth
  discussing with the advisor: these 3 subjects' underlying time-series data may be
  worth a QC pass independent of this fix.

> ⚠️ **Caveat:** at 28 subjects these numbers remain **indicative, not conclusive**
> (small folds, single random seed, multiple combinations searched). Expanding the cohort
> from 16 → 28, and separately fixing the graph-construction bug above, both acted as
> robustness checks and showed most single-run "signal" was noise — node2vec + MLP +
> Spearman is the one result that has persisted through every change. A multi-seed
> robustness check and further data collection are planned.

## Folder Structure
```
.
├── data/                              # ALL subject data is git-ignored (ADNI DUA)
│   ├── raw/{AD,CN}/                    # raw DICOM series, sorted by diagnosis
│   ├── nifti/{AD,CN}/                  # converted .nii.gz + .json sidecars (Phase 2)
│   ├── timeseries/                     # (time_points x 116 regions) matrix per subject (Phase 3)
│   │   ├── {AD,CN}/                    #   one .npy per subject
│   │   └── aal_region_labels.csv       #   116 AAL region names (column order)
│   ├── connectivity/{pearson,spearman,kendall}/{AD,CN}/   # (116 x 116) matrices (Phase 4)
│   ├── metadata/                       # ADNI per-scan XML metadata
│   └── subjects.csv                    # labels + demographics (MMSE/CDR/APOE/scanner)
├── src/                               # one script per phase (+ QC)
│   ├── graph_utils.py                  # shared graph construction (build_nx_graph)
│   ├── ggvec_bridge.py                 # subprocess bridge: nodevectors venv -> phase6
│   └── phase5e_gee.py                  # GEE model (hemisphere labels) - Phase 5e
├── outputs/
│   ├── figures/
│   │   ├── atlas_qc/                   # atlas-on-brain overlays (git-ignored: brain images)
│   │   └── connectivity/               # per-subject connectivity heatmaps
│   ├── models/                         # trained GCN (.pt) & RF (.joblib) — git-ignored
│   └── results/model_comparison.csv    # Phase 6 metrics table
├── reference/SGCN/                    # cloned reference repo — READ ONLY, git-ignored
├── requirements.txt
├── requirements-nodevectors.txt        # separate env for Phase 5d — see Setup
├── .gitignore
└── README.md
```

## Setup (macOS, Apple Silicon)
```bash
# 1. System tool for DICOM -> NIfTI conversion (Phase 2)
brew install dcm2niix

# 2. Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Python dependencies (nilearn, torch, torch-geometric, node2vec, toper, scikit-learn, ...)
pip install -r requirements.txt

# 4. Clone SGCN as a READ-ONLY architecture reference (not copied wholesale)
git clone https://github.com/Houliang-Zhou/SGCN.git reference/SGCN
```

**Note on Phase 6:** loading `torch` (GCN), `gensim`/node2vec, and `networkit`
(TopER's `GraphRicciCurvature` dependency) in the same process can segfault on macOS
due to conflicting bundled OpenMP runtimes. If `phase6_model_comparison.py` crashes
with no traceback, run it with:
```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python src/phase6_model_comparison.py
```

**Note on Phase 5d (GGVec/nodevectors):** `nodevectors`' `csrgraph` dependency calls
`nx.adj_matrix()`, removed in networkx 3.x, so it hard-pins `networkx==2.5.1` -
incompatible with node2vec's and TopER's `networkx>=3` requirement (that old networkx
in turn needs `numpy<2.0`, which needs `scipy<1.14`). This genuinely cannot share an
environment with the rest of the project, so it needs its own venv - **required for
`phase6_model_comparison.py` too**, which shells out to it via `src/ggvec_bridge.py`:
```bash
python3 -m venv ~/.venvs/nodevectors
~/.venvs/nodevectors/bin/pip install -r requirements-nodevectors.txt

# Standalone GGVec-only results (leave-one-out CV):
~/.venvs/nodevectors/bin/python src/phase5d_nodevectors.py
```

## Attribution
GCN architecture and graph construction are **inspired by** the SGCN repository
(https://github.com/Houliang-Zhou/SGCN) and the IEEE paper *"Multi-Modal Diagnosis of
Alzheimer's Disease Using Interpretable Graph Convolutional Networks"*
(https://ieeexplore.ieee.org/document/10606492). node2vec follows Grover & Leskovec (KDD
2016). TopER embeddings use the unmodified `toper` pip package
(https://github.com/AstritTola/TopER; Tola, Taiwo, Akcora & Coskunuzer, *"TopER:
Topological Embeddings in Graph Representation Learning,"* NeurIPS 2025). GGVec
embeddings use the unmodified `nodevectors` pip package
(https://github.com/VHRanger/nodevectors, Ranger). GEE embeddings use the
unmodified `gee` package (https://github.com/cshen6/GraphEmd; Shen, C., Wang, Q.,
Priebe, C.E., *"One-hot graph encoder embedding,"* IEEE TPAMI 45(6), 2023). All
pipeline code is written from scratch; inspiration is credited inline.

## Data Ethics
ADNI data is used under its Data Use Agreement. **No subject data — raw scans, NIfTI,
time-series, connectivity matrices, metadata, `subjects.csv`, or brain-image figures — is
committed to version control** (see `.gitignore`). Only code and abstract result tables
are tracked.
