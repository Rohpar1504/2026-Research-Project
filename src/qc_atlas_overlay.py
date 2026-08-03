#!/usr/bin/env python3
"""
QC (quality-control) - visual check of the AAL atlas parcellation.

Purpose:
    Confirm that the atlas "markings" land on real brain anatomy. For each subject
    it overlays the 116-region AAL atlas on the subject's own mean fMRI image, so
    you can eyeball whether the colored regions sit inside the brain and follow its
    shape. This is the honest visual check of the native-space-vs-MNI alignment
    limitation noted in Phase 3.

    It also saves one reference image of the atlas on the standard MNI template,
    which confirms the atlas itself loaded correctly (116 symmetric parcels).

What it produces (in outputs/figures/atlas_qc/):
    00_aal_on_mni_template.png   - atlas on the standard template (sanity check)
    {GROUP}_{subject}.png        - atlas overlaid on each subject's mean fMRI

This script only READS the NIfTI data and WRITES images - it changes no pipeline data.

Usage:
    .venv/bin/python src/qc_atlas_overlay.py
"""

import matplotlib
matplotlib.use("Agg")  # headless backend: save figures to file without a display

from pathlib import Path

from nilearn import datasets, image, plotting

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NIFTI_DIR = PROJECT_ROOT / "data" / "nifti"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "atlas_qc"
GROUPS = ["AD", "CN"]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Same atlas as Phase 3: classic 116-region AAL (SPM12), defined in MNI space.
    atlas = datasets.fetch_atlas_aal(version="SPM12")

    # 1) Reference: atlas on the standard MNI template. If this looks like 116
    #    symmetric colored parcels tiling a brain, the atlas loaded correctly.
    ref_path = FIG_DIR / "00_aal_on_mni_template.png"
    plotting.plot_roi(
        atlas.maps,
        title="AAL atlas (116 regions) on MNI template",
        display_mode="ortho",
        output_file=str(ref_path),
    )
    print(f"saved {ref_path.relative_to(PROJECT_ROOT)}")

    # 2) Per subject: overlay the atlas on the subject's own mean fMRI.
    for group in GROUPS:
        for nii in sorted((NIFTI_DIR / group).glob("*.nii.gz")):
            sid = nii.name.replace(".nii.gz", "")

            # Average the 4D scan over its 140 timepoints -> one 3D anatomy-like image.
            mean_img = image.mean_img(nii)

            # Match Phase 3's resampling_target="labels": put the subject's mean image
            # onto the atlas grid, so the overlay reflects exactly what the masker saw.
            mean_on_atlas = image.resample_to_img(
                mean_img, atlas.maps, interpolation="continuous"
            )

            out = FIG_DIR / f"{group}_{sid}.png"
            plotting.plot_roi(
                atlas.maps,
                bg_img=mean_on_atlas,   # subject's brain underneath
                title=f"{group}: {sid}",
                display_mode="ortho",
                alpha=0.5,              # semi-transparent regions so the brain shows through
                output_file=str(out),
            )
            print(f"saved {out.relative_to(PROJECT_ROOT)}")

    print(f"\nDone. Open the PNGs in {FIG_DIR.relative_to(PROJECT_ROOT)} to eyeball alignment.")


if __name__ == "__main__":
    main()
