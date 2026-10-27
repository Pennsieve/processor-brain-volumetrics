"""
Synthetic upstream outputs for two subjects, in the linked layout
(INPUT_DIR/<input>/...) and the flat layout (everything under INPUT_DIR,
nested). No real MRI: a 64^3, 2 mm ellipsoid "brain" with blobs carrying the
FreeSurfer/SynthSeg labels, a SynthSeg-style volumes CSV counted from that
label map, a QC CSV, WMH blobs for sub-01, a registration.json and images.json.

    python -m tests.make_fixture data/fixture --layout both
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

SHAPE = (64, 64, 64)
VOXEL = 2.0
AFFINE = np.array([[VOXEL, 0, 0, -64.0], [0, VOXEL, 0, -64.0], [0, 0, VOXEL, -64.0], [0, 0, 0, 1.0]])
CENTER = np.array([32.0, 32.0, 32.0])

# SynthSeg 2.0 volumes CSV header order (label value order), 'total intracranial' first (SynthSeg's last_first).
SYNTHSEG_NAMES = [
    (2, "left cerebral white matter"), (3, "left cerebral cortex"), (4, "left lateral ventricle"),
    (5, "left inferior lateral ventricle"), (7, "left cerebellum white matter"), (8, "left cerebellum cortex"),
    (10, "left thalamus"), (11, "left caudate"), (12, "left putamen"), (13, "left pallidum"),
    (14, "3rd ventricle"), (15, "4th ventricle"), (16, "brain-stem"), (17, "left hippocampus"),
    (18, "left amygdala"), (24, "csf"), (26, "left accumbens area"), (28, "left ventral DC"),
    (41, "right cerebral white matter"), (42, "right cerebral cortex"), (43, "right lateral ventricle"),
    (44, "right inferior lateral ventricle"), (46, "right cerebellum white matter"), (47, "right cerebellum cortex"),
    (49, "right thalamus"), (50, "right caudate"), (51, "right putamen"), (52, "right pallidum"),
    (53, "right hippocampus"), (54, "right amygdala"), (58, "right accumbens area"), (60, "right ventral DC"),
]
QC_NAMES = ["general white matter", "general grey matter", "general csf", "cerebellum", "brainstem", "thalamus",
            "putamen+pallidum", "hippocampus+amygdala"]

# Paired blobs: name -> (left label, right label, centre offset (dx, dy, dz) from CENTER, radii). Left = -x.
PAIRED = {
    "lateral_ventricle": (4, 43, (6, 2, 2), (3, 8, 3)),
    "inferior_lateral_ventricle": (5, 44, (12, 4, -8), (1.6, 1.6, 1.6)),
    "thalamus": (10, 49, (5, -2, -2), (5, 5, 5)),
    "caudate": (11, 50, (7, 8, 2), (3.5, 3.5, 3.5)),
    "putamen": (12, 51, (11, 4, -2), (4, 4, 4)),
    "pallidum": (13, 52, (9, 1, -4), (2.5, 2.5, 2.5)),
    "hippocampus": (17, 53, (13, -6, -10), (4, 6, 4)),
    "amygdala": (18, 54, (12, 0, -10), (3, 3, 3)),
    "accumbens_area": (26, 58, (4, 8, -4), (2, 2, 2)),
    "ventral_dc": (28, 60, (5, -2, -8), (3, 3, 3)),
}

SUBJECTS = {
    "sub-01": {"seed": 1, "mask_radii": (30, 31, 29), "hippo_right_radii": (4, 6, 4), "dice": 0.93, "wmh": True,
               "qc": [0.92, 0.90, 0.88, 0.91, 0.87, 0.93, 0.89, 0.90],
               "scanner": {"Manufacturer": "Siemens", "ManufacturersModelName": "Prisma", "MagneticFieldStrength": 3}},
    # sub-02: a small right hippocampus (asymmetry + low total), a poor registration and a low QC region.
    "sub-02": {"seed": 2, "mask_radii": (29, 30, 28), "hippo_right_radii": (3, 5, 3), "dice": 0.80, "wmh": False,
               "qc": [0.90, 0.88, 0.86, 0.90, 0.85, 0.91, 0.87, 0.55],
               "scanner": {"Manufacturer": "Philips", "ManufacturersModelName": "Achieva", "MagneticFieldStrength": 1.5}},
}


def _ellipsoid(center, radii) -> np.ndarray:
    g = np.indices(SHAPE).astype(float)
    d = sum(((g[i] - center[i]) / radii[i]) ** 2 for i in range(3))
    return d <= 1.0


def build_subject(spec: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(spec["seed"])
    mask = _ellipsoid(CENTER, spec["mask_radii"])
    inner = _ellipsoid(CENTER, tuple(r - 4 for r in spec["mask_radii"]))
    x = np.indices(SHAPE)[0]
    z = np.indices(SHAPE)[2]
    y = np.indices(SHAPE)[1]
    left = x < 32
    labels = np.zeros(SHAPE, dtype=np.int16)
    # cerebrum: cortex shell, white matter core
    labels[mask & left] = 3
    labels[mask & ~left] = 42
    labels[inner & left] = 2
    labels[inner & ~left] = 41
    # cerebellum: the inferior-posterior part of the mask
    cereb = mask & (z < 20) & (y < 30)
    labels[cereb & left] = 8
    labels[cereb & ~left] = 47
    cereb_wm = _ellipsoid((32, 22, 14), (9, 6, 4)) & cereb
    labels[cereb_wm & left] = 7
    labels[cereb_wm & ~left] = 46
    # brain stem: a cylinder through the middle, below the thalami
    stem = mask & ((x - 32) ** 2 + (y - 32) ** 2 <= 9) & (z >= 6) & (z < 24)
    labels[stem] = 16
    # midline CSF spaces
    labels[(x >= 31) & (x <= 33) & (y >= 29) & (y <= 36) & (z >= 27) & (z <= 31)] = 14
    labels[_ellipsoid((32, 27, 20), (2, 2, 2))] = 15
    csf = mask & ~_ellipsoid(CENTER, tuple(r - 1 for r in spec["mask_radii"])) & (z > 50)
    labels[csf] = 24
    # paired subcortical blobs
    for name, (lab_l, lab_r, (dx, dy, dz), radii) in PAIRED.items():
        radii_r = spec["hippo_right_radii"] if name == "hippocampus" else radii
        labels[_ellipsoid((32 - dx, 32 + dy, 32 + dz), radii)] = lab_l
        labels[_ellipsoid((32 + dx, 32 + dy, 32 + dz), radii_r)] = lab_r
    labels[~mask] = 0

    intensity = {0: 0, 2: 110, 41: 110, 3: 70, 42: 70, 7: 100, 46: 100, 8: 70, 47: 70, 16: 100, 4: 20, 43: 20,
                 5: 20, 44: 20, 14: 20, 15: 20, 24: 20}
    t1 = np.zeros(SHAPE, dtype=np.float32)
    for value in np.unique(labels):
        t1[labels == value] = intensity.get(int(value), 80)
    t1 += rng.normal(0, 5, SHAPE).astype(np.float32)
    t1[~mask] = 0
    t1 = np.clip(t1, 0, None).astype(np.int16)

    wmh = np.zeros(SHAPE, dtype=np.uint8)
    if spec["wmh"]:
        for c in ((22, 36, 40), (44, 30, 42), (40, 44, 38)):
            blob = _ellipsoid(c, (1.6, 1.6, 1.6)) & (labels == 2) | _ellipsoid(c, (1.6, 1.6, 1.6)) & (labels == 41)
            wmh[blob] = 1
    return {"mask": mask.astype(np.uint8), "labels": labels, "t1": t1, "wmh": wmh}


def _save(data: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, AFFINE)
    img.header.set_zooms((VOXEL, VOXEL, VOXEL))
    nib.save(img, str(path))


def _write_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def write_subject(subject: str, spec: dict, paths: dict[str, Path]) -> dict:
    """Write one subject's files to the given per-input folders. Returns the images.json entry."""
    vols = build_subject(spec)
    voxel_mm3 = VOXEL ** 3
    _save(vols["t1"], paths["brain"] / f"{subject}_desc-brain_T1w.nii.gz")
    _save(vols["mask"], paths["mask"] / f"{subject}_desc-brain_mask.nii.gz")
    _save(vols["labels"], paths["labels"] / f"{subject}_dseg.nii.gz")
    _save(vols["t1"], paths["warped"] / f"{subject}_space-MNI152NLin2009cAsym_T1w.nii.gz")

    counts = np.bincount(vols["labels"].ravel().astype(np.int64), minlength=61)
    tiv = float(vols["mask"].sum()) * voxel_mm3
    # SynthSeg's one-subject file: an empty first header cell, the subject as a path, 'total intracranial' first.
    header = [""] + ["total intracranial"] + [name for _, name in SYNTHSEG_NAMES]
    row = [f"/input/{subject}_T1w.nii.gz", f"{tiv:.3f}"] + [f"{counts[label] * voxel_mm3:.3f}" for label, _ in SYNTHSEG_NAMES]
    _write_csv(paths["volumes"] / f"{subject}_volumes.csv", [header, row])
    _write_csv(paths["qc"] / f"{subject}_qc.csv", [[""] + QC_NAMES, [f"/input/{subject}_T1w.nii.gz"] + [f"{q:.4f}" for q in spec["qc"]]])

    if spec["wmh"]:
        from scipy import ndimage

        _save(vols["wmh"], paths["lesions"] / f"{subject}_desc-wmh_mask.nii.gz")
        _save(vols["wmh"].astype(np.float32), paths["lesions"] / f"{subject}_desc-wmh_probseg.nii.gz")
        _, n = ndimage.label(vols["wmh"])
        _write_csv(paths["lesion-volumes"] / f"{subject}_lesions.csv",
                   [["subject", "lesion_volume_mm3", "lesion_count"], [subject, f"{vols['wmh'].sum() * voxel_mm3:.3f}", n]])

    paths["transforms"].mkdir(parents=True, exist_ok=True)
    (paths["transforms"] / f"{subject}_registration.json").write_text(json.dumps({
        "subject": subject, "reference": "MNI152NLin2009cAsym", "method": "antsRegistrationSyN", "dice": spec["dice"],
        "jaccard": round(spec["dice"] / (2 - spec["dice"]), 4)}, indent=2))
    paths["metadata"].mkdir(parents=True, exist_ok=True)
    sidecar = {"Modality": "MR", "RepetitionTime": 2.3, "EchoTime": 0.003, **spec["scanner"]}
    (paths["metadata"] / f"{subject}_T1w.json").write_text(json.dumps(sidecar, indent=2))
    return {"subject": subject, "file": f"t1w/{subject}_T1w.nii.gz", "modality": "T1w", "shape": list(SHAPE),
            "voxelSize": [VOXEL, VOXEL, VOXEL], "dtype": "int16", **spec["scanner"]}


def make_linked(root: Path) -> Path:
    """INPUT_DIR/<input name>/<files> for every input."""
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in ("volumes", "qc", "labels", "brain", "mask", "warped", "transforms",
                                              "lesions", "lesion-volumes", "metadata")}
    entries = [write_subject(subject, spec, paths) for subject, spec in SUBJECTS.items()]
    (paths["metadata"] / "images.json").write_text(json.dumps(entries, indent=2))
    return root


def make_flat(root: Path) -> Path:
    """Everything under INPUT_DIR, nested by subject and by step, as an un-linked run would get it."""
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for subject, spec in SUBJECTS.items():
        base = root / "derivatives" / subject
        paths = {"volumes": base / "synthseg", "qc": base / "synthseg", "labels": base / "synthseg",
                 "brain": base / "synthstrip", "mask": base / "synthstrip", "warped": base / "ants",
                 "transforms": base / "ants", "lesions": base / "lst", "lesion-volumes": base / "lst",
                 "metadata": root / "metadata"}
        entries.append(write_subject(subject, spec, paths))
    (root / "metadata" / "images.json").write_text(json.dumps({"images": entries}, indent=2))
    return root


def make_fixture(root: Path, layout: str = "both") -> dict[str, Path]:
    root = Path(root)
    out = {}
    if layout in ("linked", "both"):
        shutil.rmtree(root / "linked", ignore_errors=True)
        out["linked"] = make_linked(root / "linked")
    if layout in ("flat", "both"):
        shutil.rmtree(root / "flat", ignore_errors=True)
        out["flat"] = make_flat(root / "flat")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="folder to write <root>/linked and/or <root>/flat into")
    parser.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    args = parser.parse_args(argv)
    for name, path in make_fixture(args.root, args.layout).items():
        n = sum(1 for p in path.rglob("*") if p.is_file())
        print(f"{name}: {path} ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
