#!/usr/bin/env python3
"""
Phase 3 - Atlas parcellation & region time-series extraction.

Applies the AAL atlas to each subject's 4D NIfTI and extracts one cleaned,
averaged BOLD signal per brain region. This turns ~200,000 raw voxels into a
compact, comparable, interpretable set of ~116 region signals.

Pipeline position:
    data/nifti/{AD,CN}/<subject>.nii.gz        (4D fMRI: x, y, z, time)
        -->  [ AAL atlas + NiftiLabelsMasker ]  -->
    data/timeseries/{AD,CN}/<subject>.npy       (matrix: timepoints x regions)

Why this phase exists:
    Functional connectivity treats the brain as a NETWORK. The nodes of that
    network are brain regions - so before we can measure connectivity we must
    define the regions. The AAL atlas provides ~116 named anatomical regions;
    NiftiLabelsMasker averages each region's voxels into one signal.

Design choices (see Phase 3 proposal):
    - Signal cleaning inside the masker: detrend + standardize + band-pass
      (0.01-0.1 Hz), the standard set for resting-state connectivity.
    - Simple direct parcellation: the masker resamples the atlas onto each
      subject's grid. Anatomical alignment is approximate (scans are in native
      space, atlas is in MNI space), but the SAME parcellation is applied
      identically to every subject and every correlation method, so the
      methods comparison stays internally valid. Documented as a limitation.

Usage (must use the project venv, which has nilearn/nibabel):
    .venv/bin/python src/phase3_extract_timeseries.py

Requirements: nilearn, nibabel, numpy, pandas  (all in requirements.txt)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NIFTI_DIR = PROJECT_ROOT / "data" / "nifti"        # input:  4D NIfTI per subject
TS_DIR = PROJECT_ROOT / "data" / "timeseries"      # output: region x time matrices
GROUPS = ["AD", "CN"]

# Repetition time (seconds) - all subjects share TR ~= 3.0 s (from Phase 2 metadata).
# The band-pass filter needs this to know the sampling rate.
T_R = 3.0


def build_masker(atlas) -> NiftiLabelsMasker:
    """
    Create the masker that averages voxels within each AAL region and cleans
    the resulting signals. Cleaning options (the recommended defaults):
        detrend       - remove slow scanner drift over the ~7-min scan
        standardize   - z-score each region so regions are on a comparable scale
        low_pass=0.1  - drop fluctuations faster than 0.1 Hz (mostly noise)
        high_pass=0.01- drop fluctuations slower than 0.01 Hz (drift)
                        (low_pass + high_pass together = a 0.01-0.1 Hz band-pass,
                         the frequency band where resting-state signal lives)

    resampling_target="labels" is critical: it resamples each subject's DATA onto
    the fixed atlas grid (rather than shrinking the atlas onto each subject's grid).
    That guarantees EVERY subject yields the exact same set of regions, in the same
    column order - a hard requirement for comparing connectivity matrices and for
    feeding a GCN with a fixed node set.
    """
    return NiftiLabelsMasker(
        labels_img=atlas.maps,
        resampling_target="labels",
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=T_R,
        verbose=0,
    )


def masker_region_names(masker, atlas) -> list[str]:
    """
    Return region names aligned to the masker's actual output columns.

    After fit(), the masker exposes the integer label code for each output column
    via `region_ids_` (keys are column positions 0..n-1, plus a 'background' key).
    We map each code to its anatomical name using the atlas metadata, so the labels
    file corresponds exactly to the columns in every subject's matrix.
    """
    code_to_name = {int(code): name for code, name in zip(atlas.indices, atlas.labels)}
    ids = masker.region_ids_  # e.g. {'background': 0, 0: 2001, 1: 2002, ...}
    positions = sorted(k for k in ids if isinstance(k, (int, np.integer)))
    return [code_to_name.get(int(ids[pos]), f"region_{int(ids[pos])}") for pos in positions]


def main() -> None:
    print("Phase 3: AAL atlas parcellation -> region time-series\n")

    # Fetch the CLASSIC 116-region AAL (SPM12). nilearn's default is now AAL3v2
    # (166 regions); we pin SPM12 to match the standard used in the literature.
    atlas = datasets.fetch_atlas_aal(version="SPM12")

    # Build and fit the masker ONCE. Because resampling_target="labels" fixes the
    # region set to the atlas, fitting once gives a single consistent set of
    # columns that we then apply to every subject via transform().
    masker = build_masker(atlas)
    masker.fit()

    names = masker_region_names(masker, atlas)
    n_regions = len(names)
    print(f"AAL atlas loaded (SPM12): {n_regions} regions\n")

    # Save the region names once - Phase 7 needs them to interpret results.
    TS_DIR.mkdir(parents=True, exist_ok=True)
    labels_csv = TS_DIR / "aal_region_labels.csv"
    pd.DataFrame({"column": range(n_regions), "region_name": names}).to_csv(
        labels_csv, index=False
    )
    print(f"Saved region labels -> {labels_csv.relative_to(PROJECT_ROOT)}\n")

    total, processed = 0, 0
    shapes = set()
    for group in GROUPS:
        in_dir = NIFTI_DIR / group
        out_dir = TS_DIR / group
        out_dir.mkdir(parents=True, exist_ok=True)

        niftis = sorted(in_dir.glob("*.nii.gz"))
        print(f"[{group}] {len(niftis)} subject(s)")

        for nii in niftis:
            total += 1
            subject_id = nii.name.replace(".nii.gz", "")

            # transform() resamples THIS subject's data onto the fixed atlas grid,
            # averages voxels per region, and cleans -> (timepoints, regions).
            ts = masker.transform(str(nii))

            n_nan = int(np.isnan(ts).sum())
            out_path = out_dir / f"{subject_id}.npy"
            np.save(out_path, ts)
            processed += 1
            shapes.add(ts.shape[1])

            flag = f"   [!] contains {n_nan} NaNs" if n_nan else ""
            print(f"    {subject_id}: shape {ts.shape} -> "
                  f"{out_path.relative_to(PROJECT_ROOT)}{flag}")
        print()

    # Correctness guard: every subject MUST have the same number of regions,
    # and it must match the saved label file.
    print(f"Done: {processed}/{total} subjects processed. Regions = {n_regions}.")
    if len(shapes) != 1 or shapes.pop() != n_regions:
        sys.exit("ERROR: inconsistent region counts across subjects - matrices are not comparable.")
    if processed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
