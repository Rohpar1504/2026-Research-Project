# Does the Correlation Method Matter? Comparing Pearson, Spearman & Kendall Functional Connectivity for Alzheimer's Classification with GCNs

**Author:** Rohan Pareek — Undergraduate Researcher, Siebel School of Computing and Data Science, UIUC
**Advisor:** Prof. Pablo D. Robles-Granda

## Research Question
Does the choice of correlation method (Pearson vs. Spearman vs. Kendall) for building
brain functional connectivity matrices from fMRI meaningfully affect downstream Graph
Convolutional Network (GCN) classification of Alzheimer's Disease (AD) vs. Cognitively
Normal (CN) subjects?

## Dataset
ADNI (Alzheimer's Disease Neuroimaging Initiative), ~6–12 subjects, balanced AD/CN,
**no MCI or any other condition**. Data begins as raw DICOM fMRI scans.

## Pipeline (8 Phases)
| Phase | What it does | Main script |
|-------|--------------|-------------|
| 1 | Setup & data access (this scaffolding) | — |
| 2 | DICOM → NIfTI conversion (`dcm2niix`) | `src/phase2_dicom_to_nifti.py` |
| 3 | Apply AAL atlas, extract region time series (`NiftiLabelsMasker`) | `src/phase3_extract_timeseries.py` |
| 4 | Build 3 connectivity matrices per subject (Pearson/Spearman/Kendall) | `src/phase4_connectivity.py` |
| 5 | Connectivity → graph, train GCN + MLP classifier | `src/phase5_gcn.py` |
| 6 | Run all 3 methods, record accuracy/sensitivity/specificity | `src/phase6_experiments.py` |
| 7 | Interpret important regions/edges per method | `src/phase7_interpretation.py` |
| 8 | Write-up | `outputs/` |

Each script is meant to be **independently runnable** and reads from the previous
phase's output folder.

## Folder Structure
```
.
├── data/
│   ├── raw/{AD,CN}/          # Phase 1: raw DICOM series, sorted by diagnosis
│   ├── nifti/{AD,CN}/        # Phase 2: converted .nii.gz + .json sidecars
│   ├── timeseries/           # Phase 3: (time_points x regions) matrix per subject
│   └── connectivity/
│       ├── pearson/          # Phase 4: (regions x regions) matrix per subject
│       ├── spearman/
│       └── kendall/
├── src/                      # one script per phase
├── outputs/
│   ├── figures/              # heatmaps, comparison plots
│   ├── models/               # trained GCN checkpoints (git-ignored)
│   └── results/              # metrics tables (CSV)
├── reference/
│   └── SGCN/                 # cloned reference repo — READ ONLY, not committed
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (macOS, Apple Silicon)
```bash
# 1. System tool for DICOM -> NIfTI conversion (used in Phase 2, via terminal)
brew install dcm2niix

# 2. Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Python dependencies
pip install -r requirements.txt

# 4. Clone SGCN as a READ-ONLY architecture reference (not copied wholesale)
git clone https://github.com/Houliang-Zhou/SGCN.git reference/SGCN
```

## Attribution
GCN architecture, graph construction, and MLP classifier design are **inspired by**
the SGCN repository (https://github.com/Houliang-Zhou/SGCN) and the IEEE paper
*"Multi-Modal Diagnosis of Alzheimer's Disease Using Interpretable Graph
Convolutional Networks"* (https://ieeexplore.ieee.org/document/10606492). Code here is
written from scratch; inspiration is credited inline in the relevant scripts.

## Data Ethics
ADNI data is used under its Data Use Agreement. **No subject data is committed to
version control** (see `.gitignore`).
