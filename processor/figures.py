"""
Figures for the report: nilearn overlays (segmentation over the brain, the
MNI-warped image, the WMH mask) and matplotlib bar charts. Every function
returns the PNG's file name (relative to the page) or None when its inputs
are missing or plotting failed; a failed figure never fails the subject.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from processor.metrics import ASYMMETRY_PAIRS, PAIRS  # noqa: E402

log = logging.getLogger("processor-brain-volumetrics")

# Palette (light surface): categorical slot 1/2/3, status critical, chrome.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

BAR_STRUCTURES = ["cerebral_cortex", "cerebral_white_matter", "cerebellum_cortex", "cerebellum_white_matter",
                  "thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "accumbens_area",
                  "ventral_dc", "lateral_ventricle", "inferior_lateral_ventricle"]


def _title(text: str) -> str:
    return text.replace("_", " ")


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _bg(brain: Path | None):
    """The background for an overlay: the skull-stripped T1w when present, else none (never nilearn's
    default MNI template, since these images are in native space)."""
    return nib.load(brain) if brain else None


def segmentation(labels: Path | None, brain: Path | None, out_dir: Path, subject: str) -> str | None:
    if labels is None:
        return None
    try:
        from nilearn import plotting

        display = plotting.plot_roi(
            nib.load(labels), bg_img=_bg(brain), display_mode="ortho", title=f"{subject}: SynthSeg labels",
            draw_cross=False, alpha=0.6, cmap="gist_ncar", black_bg=False, dim=0, resampling_interpolation="nearest",
        )
        path = out_dir / "segmentation.png"
        display.savefig(path, dpi=110)
        display.close()
        return path.name
    except Exception as exc:  # noqa: BLE001 - a figure must not fail the subject
        log.warning("%s: segmentation figure failed: %s", subject, exc)
        plt.close("all")
        return None


def mni(warped: Path | None, out_dir: Path, subject: str) -> str | None:
    if warped is None:
        return None
    try:
        from nilearn import plotting

        display = plotting.plot_anat(
            nib.load(warped), display_mode="ortho", title=f"{subject}: T1w in MNI152 space (template edges)",
            draw_cross=False, black_bg=False, dim=0,
        )
        try:
            from nilearn import datasets

            display.add_edges(datasets.load_mni152_template(resolution=2), color=ORANGE)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: MNI template edges skipped: %s", subject, exc)
        path = out_dir / "mni.png"
        display.savefig(path, dpi=110)
        display.close()
        return path.name
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: MNI figure failed: %s", subject, exc)
        plt.close("all")
        return None


def lesions(lesion_mask: Path | None, brain: Path | None, out_dir: Path, subject: str) -> str | None:
    if lesion_mask is None:
        return None
    try:
        from nilearn import plotting

        img = nib.load(lesion_mask)
        data = np.asarray(img.dataobj) > 0
        if not data.any():
            log.info("%s: WMH mask is empty, no lesion figure", subject)
            return None
        cut = plotting.find_xyz_cut_coords(nib.Nifti1Image(data.astype(np.uint8), img.affine))
        display = plotting.plot_roi(
            nib.Nifti1Image(data.astype(np.uint8), img.affine), bg_img=_bg(brain), display_mode="ortho",
            cut_coords=cut, title=f"{subject}: white matter hyperintensities", draw_cross=False, alpha=0.9,
            cmap=ListedColormap([CRITICAL]), black_bg=False, dim=0, resampling_interpolation="nearest",
        )
        path = out_dir / "wmh.png"
        display.savefig(path, dpi=110)
        display.close()
        return path.name
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: lesion figure failed: %s", subject, exc)
        plt.close("all")
        return None


def volume_bars(metrics: dict[str, Any], out_dir: Path, subject: str) -> str | None:
    """Horizontal bars, left vs right, as % of ICV (mm3 when the ICV is unknown)."""
    structures = metrics["structures"]
    icv = metrics["icv"]["mm3"]
    rows = [(pair, structures.get(PAIRS[pair][0]), structures.get(PAIRS[pair][1]))
            for pair in BAR_STRUCTURES if structures.get(PAIRS[pair][0]) is not None or structures.get(PAIRS[pair][1]) is not None]
    if not rows:
        return None
    try:
        scale = (100.0 / icv) if icv else 1.0
        unit = "% of ICV" if icv else "mm³"
        names = [_title(r[0]) for r in rows]
        left = np.array([(r[1] or 0.0) * scale for r in rows])
        right = np.array([(r[2] or 0.0) * scale for r in rows])
        y = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
        h = 0.36
        ax.barh(y - h / 2 - 0.02, left, height=h, color=BLUE, label="left")
        ax.barh(y + h / 2 + 0.02, right, height=h, color=ORANGE, label="right")
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        fmt = (lambda v: f"{v:.2f}") if icv else (lambda v: f"{v:.0f}")
        for yy, l, r in zip(y, left, right):
            ax.text(l, yy - h / 2 - 0.02, " " + fmt(l), va="center", fontsize=7, color=INK)
            ax.text(r, yy + h / 2 + 0.02, " " + fmt(r), va="center", fontsize=7, color=INK)
        ax.set_xlim(0, max(left.max(), right.max()) * 1.18 or 1)
        ax.legend(frameon=False, loc="lower right")
        ax.set_title(f"{subject}: structure volumes, left vs right", loc="left", color=INK)
        return _save(fig, out_dir / "volumes.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: volume bar chart failed: %s", subject, exc)
        plt.close("all")
        return None


def _cohort_bars(subjects: list[str], series: list[tuple[str, list[float | None], str]], ylabel: str, title: str,
                 path: Path, flagged: set[str] | None = None) -> str | None:
    """Grouped bars per subject, one colour per series (fixed order), subject labels on the x axis."""
    if not subjects or all(v is None for _, values, _ in series for v in values):
        return None
    try:
        n = len(series)
        x = np.arange(len(subjects))
        width = min(0.8 / n, 0.28)
        fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(subjects) + 2.5), 3.4))
        for i, (name, values, color) in enumerate(series):
            vals = np.array([v if v is not None else np.nan for v in values], dtype=float)
            ax.bar(x - n * width / 2 + width * (i + 0.5), np.nan_to_num(vals), width=width * 0.92, color=color, label=name)
        ax.set_xticks(x, subjects, rotation=45 if len(subjects) > 6 else 0, ha="right" if len(subjects) > 6 else "center")
        ax.tick_params(axis="x", labelcolor=INK)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if n > 1:
            ax.legend(frameon=False, ncol=min(n, 4), loc="upper left", bbox_to_anchor=(0, -0.18))
        if flagged:
            tops = np.nanmax(np.array([[v if v is not None else np.nan for v in values] for _, values, _ in series], dtype=float), axis=0)
            for i, s in enumerate(subjects):
                if s in flagged and not np.isnan(tops[i]):
                    ax.text(i, tops[i], "▲ flagged", ha="center", va="bottom", color=CRITICAL, fontsize=8)
        ax.set_title(title, loc="left", color=INK)
        return _save(fig, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("cohort chart %s failed: %s", path.name, exc)
        plt.close("all")
        return None


def cohort(rows: list[dict[str, Any]], flagged: set[str], out_dir: Path) -> dict[str, str | None]:
    subjects = [r["subject"] for r in rows]
    ml = lambda key: [r[key] / 1000.0 if r.get(key) is not None else None for r in rows]  # noqa: E731
    figs = {
        "icv": _cohort_bars(subjects, [("ICV", ml("icv_mm3"), BLUE)], "mL", "Intracranial volume",
                            out_dir / "cohort_icv.png", flagged),
        "tissue": _cohort_bars(subjects, [("grey matter", [r.get("gm_pct_icv") for r in rows], BLUE),
                                          ("white matter", [r.get("wm_pct_icv") for r in rows], ORANGE),
                                          ("CSF", [r.get("csf_pct_icv") for r in rows], AQUA)],
                               "% of ICV", "Tissue fractions", out_dir / "cohort_tissue.png"),
        "hippocampus": _cohort_bars(subjects, [("left", ml("left_hippocampus_mm3"), BLUE),
                                               ("right", ml("right_hippocampus_mm3"), ORANGE)],
                                    "mL", "Hippocampal volume", out_dir / "cohort_hippocampus.png"),
        "ventricles": _cohort_bars(subjects, [("lateral ventricles", [r.get("lateral_ventricles_pct_icv") for r in rows], BLUE)],
                                   "% of ICV", "Lateral ventricles", out_dir / "cohort_ventricles.png"),
    }
    if any(r.get("wmh_volume_mm3") is not None for r in rows):
        figs["wmh"] = _cohort_bars(subjects, [("WMH", ml("wmh_volume_mm3"), BLUE)], "mL",
                                   "White matter hyperintensity volume", out_dir / "cohort_wmh.png")
    return figs


__all__ = ["segmentation", "mni", "lesions", "volume_bars", "cohort", "ASYMMETRY_PAIRS"]
