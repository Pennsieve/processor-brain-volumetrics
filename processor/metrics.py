"""
Per-subject volumetric metrics: structure volumes (from a SynthSeg volumes CSV
or counted from a label map), intracranial volume, tissue totals, % of ICV,
left/right asymmetry, white matter hyperintensities, the registration check,
SynthSeg QC scores, and the screening flags.

Everything is optional: a metric whose inputs are missing is None, and the
flags only test metrics that are known.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

# SynthSeg / FreeSurfer label scheme: label value -> (SynthSeg CSV column name, our column key).
STRUCTURES: list[tuple[int, str, str]] = [
    (2, "left cerebral white matter", "left_cerebral_white_matter"),
    (3, "left cerebral cortex", "left_cerebral_cortex"),
    (4, "left lateral ventricle", "left_lateral_ventricle"),
    (5, "left inferior lateral ventricle", "left_inferior_lateral_ventricle"),
    (7, "left cerebellum white matter", "left_cerebellum_white_matter"),
    (8, "left cerebellum cortex", "left_cerebellum_cortex"),
    (10, "left thalamus", "left_thalamus"),
    (11, "left caudate", "left_caudate"),
    (12, "left putamen", "left_putamen"),
    (13, "left pallidum", "left_pallidum"),
    (14, "3rd ventricle", "third_ventricle"),
    (15, "4th ventricle", "fourth_ventricle"),
    (16, "brain-stem", "brain_stem"),
    (17, "left hippocampus", "left_hippocampus"),
    (18, "left amygdala", "left_amygdala"),
    (24, "CSF", "csf"),
    (26, "left accumbens area", "left_accumbens_area"),
    (28, "left ventral DC", "left_ventral_dc"),
    (41, "right cerebral white matter", "right_cerebral_white_matter"),
    (42, "right cerebral cortex", "right_cerebral_cortex"),
    (43, "right lateral ventricle", "right_lateral_ventricle"),
    (44, "right inferior lateral ventricle", "right_inferior_lateral_ventricle"),
    (46, "right cerebellum white matter", "right_cerebellum_white_matter"),
    (47, "right cerebellum cortex", "right_cerebellum_cortex"),
    (49, "right thalamus", "right_thalamus"),
    (50, "right caudate", "right_caudate"),
    (51, "right putamen", "right_putamen"),
    (52, "right pallidum", "right_pallidum"),
    (53, "right hippocampus", "right_hippocampus"),
    (54, "right amygdala", "right_amygdala"),
    (58, "right accumbens area", "right_accumbens_area"),
    (60, "right ventral DC", "right_ventral_dc"),
]
STRUCTURE_KEYS = [key for _, _, key in STRUCTURES]
LABEL_TO_KEY = {label: key for label, _, key in STRUCTURES}
KEY_TO_LABEL = {key: label for label, _, key in STRUCTURES}
KEY_TO_NAME = {key: name for _, name, key in STRUCTURES}

# Column-name aliases, after normalisation (lower case, non-alphanumerics -> '_').
COLUMN_ALIASES = {
    "3rd_ventricle": "third_ventricle",
    "4th_ventricle": "fourth_ventricle",
    "brainstem": "brain_stem",
    "brain_stem": "brain_stem",
    "total_intracranial": "total_intracranial",
    "tiv": "total_intracranial",
    "etiv": "total_intracranial",
    "estimatedtotalintracranialvol": "total_intracranial",
}

# Bilateral structures: display name -> (left key, right key).
PAIRS: dict[str, tuple[str, str]] = {
    "cerebral_white_matter": ("left_cerebral_white_matter", "right_cerebral_white_matter"),
    "cerebral_cortex": ("left_cerebral_cortex", "right_cerebral_cortex"),
    "lateral_ventricle": ("left_lateral_ventricle", "right_lateral_ventricle"),
    "inferior_lateral_ventricle": ("left_inferior_lateral_ventricle", "right_inferior_lateral_ventricle"),
    "cerebellum_white_matter": ("left_cerebellum_white_matter", "right_cerebellum_white_matter"),
    "cerebellum_cortex": ("left_cerebellum_cortex", "right_cerebellum_cortex"),
    "thalamus": ("left_thalamus", "right_thalamus"),
    "caudate": ("left_caudate", "right_caudate"),
    "putamen": ("left_putamen", "right_putamen"),
    "pallidum": ("left_pallidum", "right_pallidum"),
    "hippocampus": ("left_hippocampus", "right_hippocampus"),
    "amygdala": ("left_amygdala", "right_amygdala"),
    "accumbens_area": ("left_accumbens_area", "right_accumbens_area"),
    "ventral_dc": ("left_ventral_dc", "right_ventral_dc"),
}
ASYMMETRY_PAIRS = ["hippocampus", "amygdala", "thalamus", "caudate", "putamen", "pallidum", "lateral_ventricle"]
PCT_PAIRS = ["hippocampus", "amygdala", "thalamus", "caudate", "putamen", "pallidum"]

# Tissue classes (sums over structure keys).
GM_KEYS = [
    "left_cerebral_cortex", "right_cerebral_cortex",
    "left_cerebellum_cortex", "right_cerebellum_cortex",
    "left_thalamus", "right_thalamus", "left_caudate", "right_caudate",
    "left_putamen", "right_putamen", "left_pallidum", "right_pallidum",
    "left_hippocampus", "right_hippocampus", "left_amygdala", "right_amygdala",
    "left_accumbens_area", "right_accumbens_area", "left_ventral_dc", "right_ventral_dc",
]
WM_KEYS = [
    "left_cerebral_white_matter", "right_cerebral_white_matter",
    "left_cerebellum_white_matter", "right_cerebellum_white_matter",
]
CSF_KEYS = [
    "left_lateral_ventricle", "right_lateral_ventricle",
    "left_inferior_lateral_ventricle", "right_inferior_lateral_ventricle",
    "third_ventricle", "fourth_ventricle", "csf",
]
CEREBELLUM_KEYS = [
    "left_cerebellum_white_matter", "right_cerebellum_white_matter",
    "left_cerebellum_cortex", "right_cerebellum_cortex",
]

# SynthSeg --qc regions (SynthSeg 2.0 names), normalised.
QC_REGIONS = [
    "general_white_matter", "general_grey_matter", "general_csf", "cerebellum",
    "brainstem", "thalamus", "putamen_pallidum", "hippocampus_amygdala",
]

# Screening heuristics, not clinical cut-offs (see README). FLAG_THRESHOLDS overrides these.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "icv_min_ml": 1000.0,
    "icv_max_ml": 2000.0,
    "ai_hippocampus_max": 0.15,
    "lateral_ventricles_pct_icv_max": 5.0,
    "hippocampus_total_min_ml": 5.0,
    "registration_dice_min": 0.85,
    "synthseg_qc_min": 0.65,
}

# Columns the cohort summary (mean/sd) covers.
DERIVED_COLUMNS = [
    "icv_mm3", "gm_pct_icv", "wm_pct_icv", "csf_pct_icv",
    "lateral_ventricles_pct_icv", "hippocampus_pct_icv", "amygdala_pct_icv", "thalamus_pct_icv",
    "caudate_pct_icv", "putamen_pct_icv", "pallidum_pct_icv", "cerebellum_pct_icv",
    "ai_hippocampus", "ai_amygdala", "ai_thalamus", "ai_caudate", "ai_putamen", "ai_pallidum",
    "ai_lateral_ventricle",
    "wmh_volume_mm3", "wmh_count", "wmh_pct_wm", "registration_dice", "synthseg_qc_min",
]


def normalise_column(name: Any) -> str:
    """'Left Cerebral White Matter' -> 'left_cerebral_white_matter'; '3rd ventricle' -> 'third_ventricle'."""
    text = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if text.startswith("unnamed"):
        return ""
    return COLUMN_ALIASES.get(text, text)


def to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def parse_thresholds(text: str | None) -> dict[str, float]:
    """DEFAULT_THRESHOLDS overridden by the FLAG_THRESHOLDS JSON object (unknown keys are kept, so a
    future check can read them; non-numeric values are rejected)."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    if text and text.strip():
        override = json.loads(text)
        if not isinstance(override, dict):
            raise ValueError("FLAG_THRESHOLDS must be a JSON object")
        for key, value in override.items():
            f = to_float(value)
            if f is None:
                raise ValueError(f"FLAG_THRESHOLDS[{key!r}] is not a number: {value!r}")
            thresholds[str(key)] = f
    return thresholds


# --- volumes ---------------------------------------------------------------------------------------------

def structures_from_row(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Known structure volumes (by our key) and the other numeric columns (e.g. total_intracranial) from a
    SynthSeg CSV row whose keys are already normalised."""
    known: dict[str, float] = {}
    extra: dict[str, float] = {}
    for col, value in row.items():
        f = to_float(value)
        if f is None or not col or col == "subject":
            continue
        if col in KEY_TO_LABEL:
            known[col] = f
        else:
            extra[col] = f
    return known, extra


def structures_from_labels(label_img) -> tuple[dict[str, float], float, float]:
    """Structure volumes counted from a label map (voxels x voxel volume, in mm3), plus the voxel volume and
    the volume of every non-zero label (a brain-volume stand-in for ICV)."""
    data = np.asarray(label_img.dataobj)
    data = np.rint(data).astype(np.int64)
    data[data < 0] = 0
    voxel_mm3 = float(np.prod(label_img.header.get_zooms()[:3]))
    counts = np.bincount(data.ravel(), minlength=max(KEY_TO_LABEL.values()) + 1)
    volumes = {key: float(counts[label]) * voxel_mm3 for label, key in LABEL_TO_KEY.items() if counts[label] > 0}
    nonzero = float(counts[1:].sum()) * voxel_mm3
    return volumes, voxel_mm3, nonzero


def mask_volume_mm3(mask_img) -> float:
    data = np.asarray(mask_img.dataobj)
    voxel_mm3 = float(np.prod(mask_img.header.get_zooms()[:3]))
    return float(np.count_nonzero(data > 0)) * voxel_mm3


def lesion_stats_from_mask(mask_img) -> tuple[float, int]:
    """(volume mm3, connected-component count) of a binary lesion mask."""
    from scipy import ndimage

    data = np.asarray(mask_img.dataobj) > 0
    voxel_mm3 = float(np.prod(mask_img.header.get_zooms()[:3]))
    _, count = ndimage.label(data)
    return float(np.count_nonzero(data)) * voxel_mm3, int(count)


def find_dice(obj: Any) -> float | None:
    """The first numeric value under a key named 'dice' (any case) in a nested JSON object."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() == "dice":
                f = to_float(value)
                if f is not None:
                    return f
        for value in obj.values():
            f = find_dice(value)
            if f is not None:
                return f
    elif isinstance(obj, list):
        for value in obj:
            f = find_dice(value)
            if f is not None:
                return f
    return None


# --- derived metrics ---------------------------------------------------------------------------------------

def _sum(volumes: dict[str, float], keys: list[str]) -> float | None:
    present = [volumes[k] for k in keys if k in volumes]
    return float(sum(present)) if present else None


def _pct(part: float | None, whole: float | None) -> float | None:
    if part is None or not whole:
        return None
    return 100.0 * part / whole


def _asymmetry(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or (left + right) == 0:
        return None
    return 2.0 * (left - right) / (left + right)


def derive(
    subject: str,
    structures: dict[str, float],
    volumes_source: str | None,
    icv_mm3: float | None,
    icv_source: str | None,
    wmh: dict[str, Any] | None,
    registration_dice: float | None,
    qc: dict[str, float] | None,
) -> dict[str, Any]:
    """The nested per-subject metrics block. All volumes in mm3."""
    totals: dict[str, float | None] = {
        "gm": _sum(structures, GM_KEYS),
        "wm": _sum(structures, WM_KEYS),
        "csf": _sum(structures, CSF_KEYS),
        "cerebellum": _sum(structures, CEREBELLUM_KEYS),
        "brain_stem": structures.get("brain_stem"),
    }
    for pair, (left, right) in PAIRS.items():
        totals[pair] = _sum(structures, [left, right])
    pct: dict[str, float | None] = {
        "gm": _pct(totals["gm"], icv_mm3),
        "wm": _pct(totals["wm"], icv_mm3),
        "csf": _pct(totals["csf"], icv_mm3),
        "lateral_ventricles": _pct(totals["lateral_ventricle"], icv_mm3),
        "cerebellum": _pct(totals["cerebellum"], icv_mm3),
    }
    for pair in PCT_PAIRS:
        pct[pair] = _pct(totals[pair], icv_mm3)
    asymmetry = {
        pair: _asymmetry(structures.get(PAIRS[pair][0]), structures.get(PAIRS[pair][1])) for pair in ASYMMETRY_PAIRS
    }
    if wmh is not None:
        wmh = dict(wmh)
        wmh["pct_wm"] = _pct(wmh.get("volume_mm3"), totals["wm"])
    qc_block = None
    if qc:
        qc_block = dict(qc)
        qc_block["min"] = min(qc.values())
    return {
        "subject": subject,
        "icv": {"mm3": icv_mm3, "ml": icv_mm3 / 1000.0 if icv_mm3 is not None else None, "source": icv_source},
        "volumes_source": volumes_source,
        "structures": {key: structures.get(key) for key in STRUCTURE_KEYS},
        "totals": totals,
        "pct_icv": pct,
        "asymmetry": asymmetry,
        "wmh": wmh,
        "registration": {"dice": registration_dice} if registration_dice is not None else None,
        "qc": qc_block,
    }


def flags_for(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Screening flags: checks whose value lies outside the plausible range. Unknown values are not flagged."""
    flags: list[dict[str, Any]] = []

    def flag(check: str, value: float, threshold: float, message: str) -> None:
        flags.append({"check": check, "value": value, "threshold": threshold, "message": message})

    icv_ml = metrics["icv"]["ml"]
    if icv_ml is not None:
        lo, hi = thresholds["icv_min_ml"], thresholds["icv_max_ml"]
        if icv_ml < lo:
            flag("icv_low", icv_ml, lo, f"ICV {icv_ml:.0f} mL is below {lo:.0f} mL")
        elif icv_ml > hi:
            flag("icv_high", icv_ml, hi, f"ICV {icv_ml:.0f} mL is above {hi:.0f} mL")
    ai = metrics["asymmetry"].get("hippocampus")
    if ai is not None and abs(ai) > thresholds["ai_hippocampus_max"]:
        flag("ai_hippocampus", ai, thresholds["ai_hippocampus_max"],
             f"hippocampal asymmetry index {ai:+.2f} exceeds ±{thresholds['ai_hippocampus_max']:.2f}")
    lv = metrics["pct_icv"].get("lateral_ventricles")
    if lv is not None and lv > thresholds["lateral_ventricles_pct_icv_max"]:
        flag("lateral_ventricles_pct_icv", lv, thresholds["lateral_ventricles_pct_icv_max"],
             f"lateral ventricles are {lv:.1f}% of ICV (above {thresholds['lateral_ventricles_pct_icv_max']:.1f}%)")
    hippo = metrics["totals"].get("hippocampus")
    if hippo is not None and hippo / 1000.0 < thresholds["hippocampus_total_min_ml"]:
        flag("hippocampus_total", hippo / 1000.0, thresholds["hippocampus_total_min_ml"],
             f"total hippocampal volume {hippo / 1000.0:.2f} mL is below {thresholds['hippocampus_total_min_ml']:.1f} mL")
    reg = metrics.get("registration") or {}
    dice = reg.get("dice")
    if dice is not None and dice < thresholds["registration_dice_min"]:
        flag("registration_dice", dice, thresholds["registration_dice_min"],
             f"MNI registration Dice {dice:.2f} is below {thresholds['registration_dice_min']:.2f}")
    qc = metrics.get("qc") or {}
    for region, score in qc.items():
        if region != "min" and score is not None and score < thresholds["synthseg_qc_min"]:
            flag(f"synthseg_qc_{region}", score, thresholds["synthseg_qc_min"],
                 f"SynthSeg QC for {region.replace('_', ' ')} is {score:.2f} (below {thresholds['synthseg_qc_min']:.2f})")
    return flags


def flatten(metrics: dict[str, Any], flags: list[dict[str, Any]]) -> dict[str, Any]:
    """One CSV row for tables/volumes.csv."""
    row: dict[str, Any] = {
        "subject": metrics["subject"],
        "icv_mm3": metrics["icv"]["mm3"],
        "icv_source": metrics["icv"]["source"],
        "volumes_source": metrics["volumes_source"],
    }
    for key in STRUCTURE_KEYS:
        row[f"{key}_mm3"] = metrics["structures"].get(key)
    totals = metrics["totals"]
    row["gm_total_mm3"] = totals["gm"]
    row["wm_total_mm3"] = totals["wm"]
    row["csf_total_mm3"] = totals["csf"]
    row["brain_stem_total_mm3"] = totals["brain_stem"]
    for pair in PAIRS:
        row[f"{pair}_total_mm3"] = totals[pair]
    pct = metrics["pct_icv"]
    row["gm_pct_icv"] = pct["gm"]
    row["wm_pct_icv"] = pct["wm"]
    row["csf_pct_icv"] = pct["csf"]
    row["lateral_ventricles_pct_icv"] = pct["lateral_ventricles"]
    for pair in PCT_PAIRS:
        row[f"{pair}_pct_icv"] = pct[pair]
    row["cerebellum_pct_icv"] = pct["cerebellum"]
    for pair in ASYMMETRY_PAIRS:
        row[f"ai_{pair}"] = metrics["asymmetry"][pair]
    wmh = metrics.get("wmh") or {}
    row["wmh_volume_mm3"] = wmh.get("volume_mm3")
    row["wmh_count"] = wmh.get("count")
    row["wmh_pct_wm"] = wmh.get("pct_wm")
    row["registration_dice"] = (metrics.get("registration") or {}).get("dice")
    qc = metrics.get("qc") or {}
    row["synthseg_qc_min"] = qc.get("min")
    for region in QC_REGIONS:
        row[f"qc_{region}"] = qc.get(region)
    for region in qc:
        if region != "min" and region not in QC_REGIONS:
            row[f"qc_{region}"] = qc[region]
    row["n_flags"] = len(flags)
    row["flags"] = ";".join(f["check"] for f in flags)
    return row


def cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """n, and mean/sd/min/max per derived column over the subjects that have it."""
    summary: dict[str, Any] = {"n": len(rows), "columns": {}}
    for col in DERIVED_COLUMNS:
        values = [r[col] for r in rows if r.get(col) is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary["columns"][col] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def version_of(repo_root: Path) -> str:
    try:
        return (repo_root / "VERSION").read_text().strip() or "0.0.0"
    except OSError:
        return "0.0.0"
