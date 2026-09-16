# Does the Correlation Method Matter? Pearson vs. Spearman vs. Kendall Functional Connectivity for Alzheimer's Classification

**Author:** Rohan Pareek — Undergraduate Researcher, Siebel School of Computing and Data Science, UIUC
**Advisor:** Prof. Pablo D. Robles-Granda

## Research Question
Does the choice of correlation method (Pearson vs. Spearman vs. Kendall) for building
brain functional-connectivity networks from fMRI meaningfully affect downstream
classification of Alzheimer's Disease (AD) vs. Cognitively Normal (CN) subjects?
The comparison is evaluated across several graph-based classification models.

## Dataset
ADNI (Alzheimer's Disease Neuroimaging Initiative) — **16 subjects, balanced 8 AD / 8 CN**,
all **ADNI-2 resting-state fMRI** (Philips 3T, TR ≈ 3 s, "Resting State fMRI" protocol).
Groups are **sex-mirrored** (each 5M/3F) and **age-matched** (AD ≈ 75.8, CN ≈ 77.7).
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
| 6 | 4-fold CV comparison: models × methods × metrics | `src/phase6_model_comparison.py` | ✅ |
| 7 | Interpret important regions/edges per method | (planned) | ⏭️ |
| 8 | Write-up | `outputs/` | ⏭️ |

Each script is **independently runnable** and reads from the previous phase's output folder.
Run scripts with the project venv, e.g. `.venv/bin/python src/phase4_connectivity.py`.

## Models Compared
Two families of graph-based models classify each subject's connectivity network:

- **GCN** (`phase5_gcn.py`) — a Graph Convolutional Network + MLP head that learns
  directly from the graph structure. *Data-hungry; collapses to one class at n=16.*
- **node2vec + classifier** (`phase5b`, `phase6`) — node2vec embeds each graph's
  structure (unsupervised), mean-pooled into a per-subject vector, then classified by
  **Random Forest**, **MLP**, or **Logistic Regression** (a sigmoid shallow classifier).

All models see identical graphs (top-20% strongest edges by `|correlation|`) so any
performance difference reflects the model or the connectivity method, not preprocessing.

## Current Status & Findings (indicative, n = 16)
Phase 6 evaluates every model × method with **4-fold stratified cross-validation**,
reporting accuracy, sensitivity, specificity, and F1 (`outputs/results/model_comparison.csv`).

- **Best result:** node2vec + **MLP** on **Spearman** — accuracy **0.875**, F1 **0.875**.
- **MLP and Logistic Regression** beat Random Forest and the GCN — the first
  above-chance signal that AD vs. CN is separable from connectivity.
- For the working models, **rank-based methods (Spearman, Kendall) outperform Pearson**.
- **The GCN still collapses** (sensitivity ≈ 0.125 — predicts almost everyone CN),
  consistent with being too data-hungry at this sample size.

> ⚠️ **Caveat:** with 16 subjects these numbers are **indicative, not conclusive**
> (small folds, single random seed, many combinations searched). The *trends* are more
> reliable than any single value; validation with more seeds and more subjects is planned.

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
├── outputs/
│   ├── figures/
│   │   ├── atlas_qc/                   # atlas-on-brain overlays (git-ignored: brain images)
│   │   └── connectivity/               # per-subject connectivity heatmaps
│   ├── models/                         # trained GCN (.pt) & RF (.joblib) — git-ignored
│   └── results/model_comparison.csv    # Phase 6 metrics table
├── reference/SGCN/                    # cloned reference repo — READ ONLY, git-ignored
├── requirements.txt
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

# 3. Python dependencies (nilearn, torch, torch-geometric, node2vec, scikit-learn, ...)
pip install -r requirements.txt

# 4. Clone SGCN as a READ-ONLY architecture reference (not copied wholesale)
git clone https://github.com/Houliang-Zhou/SGCN.git reference/SGCN
```

## Attribution
GCN architecture and graph construction are **inspired by** the SGCN repository
(https://github.com/Houliang-Zhou/SGCN) and the IEEE paper *"Multi-Modal Diagnosis of
Alzheimer's Disease Using Interpretable Graph Convolutional Networks"*
(https://ieeexplore.ieee.org/document/10606492). node2vec follows Grover & Leskovec (KDD
2016). All pipeline code is written from scratch; inspiration is credited inline.

## Data Ethics
ADNI data is used under its Data Use Agreement. **No subject data — raw scans, NIfTI,
time-series, connectivity matrices, metadata, `subjects.csv`, or brain-image figures — is
committed to version control** (see `.gitignore`). Only code and abstract result tables
are tracked.
