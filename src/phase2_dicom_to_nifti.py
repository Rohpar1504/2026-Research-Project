#!/usr/bin/env python3
"""
Phase 2 - DICOM to NIfTI conversion.

Converts each subject's raw DICOM series (thousands of single-slice .dcm files)
into a single compressed 4D NIfTI (.nii.gz) using the `dcm2niix` command-line tool.

Pipeline position:
    data/raw/{AD,CN}/<subject>/...   (DICOM, ~6720 files each)
        -->  [ this script runs dcm2niix ]  -->
    data/nifti/{AD,CN}/<subject>.nii.gz   (one clean 4D NIfTI, ready for Phase 3)

Design note:
    The actual conversion is performed by dcm2niix (Chris Rorden). This script only
    ORCHESTRATES it - locating each subject, giving outputs clean/predictable names,
    and verifying the result. This matches the guidance that conversion is done by the
    dcm2niix tool rather than reimplemented in Python, while still fitting the project's
    "one runnable script per phase" structure.

Usage:
    python src/phase2_dicom_to_nifti.py

Requirements:
    - dcm2niix on PATH            (brew install dcm2niix)
    - nibabel (optional)         (only used to print output dimensions)
"""

from pathlib import Path
import shutil
import subprocess
import sys

# --- Project paths -----------------------------------------------------------
# This script lives in <project>/src/, so the project root is one directory up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"       # input:  DICOM, sorted by diagnosis
NIFTI_DIR = PROJECT_ROOT / "data" / "nifti"   # output: NIfTI, mirrors the AD/CN split
GROUPS = ["AD", "CN"]                         # diagnosis labels = folder names


def check_dcm2niix() -> None:
    """Exit early with a clear message if dcm2niix is not installed."""
    if shutil.which("dcm2niix") is None:
        sys.exit(
            "ERROR: dcm2niix not found on PATH.\n"
            "Install it with:  brew install dcm2niix"
        )


def convert_subject(subject_dir: Path, out_dir: Path) -> Path | None:
    """
    Convert one subject's DICOM folder into a single .nii.gz inside out_dir.

    Returns the path to the created NIfTI file, or None if conversion produced nothing.
    """
    subject_id = subject_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # dcm2niix options:
    #   -z y            compress the output to .nii.gz (smaller, standard for research)
    #   -f <subject_id> name the output file after the subject (clean, predictable)
    #   -o <out_dir>    directory to write the output into
    #   -b y            also write the BIDS .json sidecar (records TR, slice timing, etc.)
    #   <subject_dir>   input folder; dcm2niix recursively searches its subfolders,
    #                   reads the DICOM headers, and reassembles slices/timepoints in order
    cmd = [
        "dcm2niix",
        "-z", "y",
        "-f", subject_id,
        "-o", str(out_dir),
        "-b", "y",
        str(subject_dir),
    ]

    print(f"  Converting {subject_id} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  !! dcm2niix failed for {subject_id}")
        print("  " + (result.stderr.strip() or result.stdout.strip()))
        return None

    # dcm2niix occasionally appends a suffix to the filename; match whatever it wrote.
    outputs = sorted(out_dir.glob(f"{subject_id}*.nii.gz"))
    return outputs[0] if outputs else None


def describe(nifti_path: Path) -> str:
    """Return a short description of the NIfTI (its 4D shape if nibabel is available)."""
    try:
        import nibabel as nib
        img = nib.load(str(nifti_path))
        # shape is (x, y, z, timepoints) for a 4D fMRI volume
        return f"shape = {img.shape}"
    except ImportError:
        size_mb = nifti_path.stat().st_size / 1e6
        return f"{size_mb:.1f} MB (install nibabel to see dimensions)"


def main() -> None:
    check_dcm2niix()
    print("Phase 2: DICOM -> NIfTI conversion\n")

    total, converted = 0, 0
    for group in GROUPS:
        group_raw = RAW_DIR / group
        group_out = NIFTI_DIR / group

        if not group_raw.is_dir():
            print(f"[{group}] no folder at {group_raw} - skipping\n")
            continue

        subjects = sorted(p for p in group_raw.iterdir() if p.is_dir())
        print(f"[{group}] {len(subjects)} subject(s)")

        for subject_dir in subjects:
            total += 1
            nifti = convert_subject(subject_dir, group_out)
            if nifti is not None:
                converted += 1
                rel = nifti.relative_to(PROJECT_ROOT)
                print(f"    -> {rel}  ({describe(nifti)})")
        print()

    print(f"Done: {converted}/{total} subjects converted.")
    # Non-zero exit if any subject failed, so problems are easy to notice.
    if converted != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
