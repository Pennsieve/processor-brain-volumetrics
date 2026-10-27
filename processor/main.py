"""
processor-brain-volumetrics: the report step of the Clinical Brain MRI
Volumetrics workflow. It gathers what the upstream steps produced (SynthSeg
volumes/QC/labels, the skull-stripped T1w and brain mask, the MNI-warped T1w
and registration check, WMH masks and lesion tables, image metadata), works
out per-subject volumetrics, and writes

  OUTPUT_DIR/report/index.html             cohort page
  OUTPUT_DIR/report/<sub>/report.html      per-subject page + PNG figures
  OUTPUT_DIR/report/provenance.json
  OUTPUT_DIR/tables/volumes.csv            one row per subject
  OUTPUT_DIR/tables/metrics.json           nested metrics, cohort summary, flags

Every input is optional; a subject needs a volumes CSV or a label map to get
a report. Subjects are matched across inputs by their BIDS subject label.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("processor-brain-volumetrics")
# The Lambda runtime configures the root logger before basicConfig can, at
# WARNING, so the per-subject INFO lines would be lost without this.
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "processor-brain-volumetrics"
OUTPUTS = ("report", "tables")
PARAMETERS = ("REPORT_TITLE", "FLAG_THRESHOLDS")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def process_subject(inputs, thresholds: dict[str, float], report_dir: Path, input_dir: Path) -> dict[str, Any]:
    """Metrics, flags, figures and the page for one subject. Raises on failure."""
    import nibabel as nib

    from processor import figures, metrics as M

    subject = inputs.subject
    structures: dict[str, float] = {}
    extra: dict[str, float] = {}
    volumes_source = None
    label_img = None
    labels_nonzero = None
    if inputs.volumes_row is not None:
        structures, extra = M.structures_from_row(inputs.volumes_row)
        volumes_source = "volumes_csv"
    if (not structures) and inputs.labels is not None:
        label_img = nib.load(inputs.labels)
        structures, _, labels_nonzero = M.structures_from_labels(label_img)
        volumes_source = "labels"
    if not structures:
        raise ValueError("no structure volumes: the volumes CSV has no known columns and there is no label map")

    icv = None
    icv_source = None
    if inputs.mask is not None:
        icv = M.mask_volume_mm3(nib.load(inputs.mask))
        icv_source = "brain_mask"
    if not icv and extra.get("total_intracranial"):
        icv, icv_source = extra["total_intracranial"], "synthseg_total_intracranial"
    if not icv and inputs.labels is not None:
        if labels_nonzero is None:
            labels_nonzero = M.structures_from_labels(nib.load(inputs.labels))[2]
        icv, icv_source = labels_nonzero, "labelled_voxels"
        log.warning("%s: no brain mask and no total intracranial volume; using the labelled volume as ICV", subject)
    if not icv:
        icv, icv_source = None, None

    wmh = None
    if inputs.lesion_row is not None and inputs.lesion_row.get("volume_mm3") is not None:
        wmh = {"volume_mm3": inputs.lesion_row["volume_mm3"], "count": inputs.lesion_row.get("count"), "source": "lesions_csv"}
    elif inputs.lesion_mask is not None:
        volume, count = M.lesion_stats_from_mask(nib.load(inputs.lesion_mask))
        wmh = {"volume_mm3": volume, "count": count, "source": "wmh_mask"}

    dice = None
    if inputs.registration is not None:
        try:
            with open(inputs.registration) as fh:
                dice = M.find_dice(json.load(fh))
        except (OSError, ValueError) as exc:
            log.warning("%s: cannot read %s: %s", subject, inputs.registration, exc)

    qc = None
    if inputs.qc_row is not None:
        qc = {col: f for col, value in inputs.qc_row.items() if col != "subject" and (f := M.to_float(value)) is not None}
        qc = qc or None

    m = M.derive(subject, structures, volumes_source, icv, icv_source, wmh, dice, qc)
    flags = M.flags_for(m, thresholds)

    sub_dir = report_dir / subject
    sub_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "segmentation": figures.segmentation(inputs.labels, inputs.brain, sub_dir, subject),
        "volumes": figures.volume_bars(m, sub_dir, subject),
        "mni": figures.mni(inputs.warped, sub_dir, subject),
        "lesions": figures.lesions(inputs.lesion_mask, inputs.brain, sub_dir, subject),
    }
    input_paths = {
        "volumes": _rel(inputs.volumes_file, input_dir), "qc": _rel(inputs.qc_file, input_dir),
        "labels": _rel(inputs.labels, input_dir), "brain": _rel(inputs.brain, input_dir),
        "mask": _rel(inputs.mask, input_dir), "warped": _rel(inputs.warped, input_dir),
        "transforms": _rel(inputs.registration, input_dir), "lesions": _rel(inputs.lesion_mask, input_dir),
        "lesion-volumes": _rel(inputs.lesion_file, input_dir),
        "metadata": ", ".join(_rel(p, input_dir) for p in inputs.metadata_files) or None,
    }
    return {
        "subject": subject, "metrics": m, "flags": flags, "figures": figs,
        "inputs": {k: v for k, v in input_paths.items() if v}, "metadata": inputs.metadata, "dir": sub_dir,
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    input_dir = os.environ.get("INPUT_DIR")
    output_dir = os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    from processor import bids_names, figures, inputs as I, metrics as M, report as R

    title = os.environ.get("REPORT_TITLE") or "Clinical Brain MRI Volumetrics"
    try:
        thresholds = M.parse_thresholds(os.environ.get("FLAG_THRESHOLDS"))
    except ValueError as exc:
        log.error("bad FLAG_THRESHOLDS: %s", exc)
        return 2
    parameters = {"REPORT_TITLE": title, "FLAG_THRESHOLDS": os.environ.get("FLAG_THRESHOLDS", "{}")}

    report_dir = bids_names.output_dir_for(output_dir, list(OUTPUTS), "report")
    tables_dir = bids_names.output_dir_for(output_dir, list(OUTPUTS), "tables")

    subjects, found = I.gather(input_dir)
    for name in I.INPUT_NAMES:
        files = found.get(name, [])
        log.info("input %s: %d file(s)%s", name, len(files), (": " + ", ".join(_rel(f, input_dir) for f in files[:6]) +
                                                              (" …" if len(files) > 6 else "")) if files else "")
    reportable = {s: v for s, v in subjects.items() if v.reportable}
    if not reportable:
        listing = [str(p.relative_to(input_dir)) for p in sorted(input_dir.rglob("*")) if p.is_file()]
        log.error("no subject has a volumes CSV (*_volumes.csv) or a label map (*_dseg.nii.gz); INPUT_DIR holds %d file(s): %s",
                  len(listing), ", ".join(listing[:40]) + (" …" if len(listing) > 40 else ""))
        return 1
    for s in subjects:
        if s not in reportable:
            log.warning("%s: only side inputs (no volumes CSV, no label map); skipped", s)

    import nilearn

    env = R.environment()
    app_version = M.version_of(REPO_ROOT)
    generated_at = started.isoformat(timespec="seconds")
    results: list[dict[str, Any]] = []
    skipped = [{"subject": s, "reason": "no volumes CSV and no label map"} for s in subjects if s not in reportable]
    provenance_subjects = []
    rows = []
    for subject, inputs in reportable.items():
        t0 = time.monotonic()
        try:
            result = process_subject(inputs, thresholds, report_dir, input_dir)
            row = M.flatten(result["metrics"], result["flags"])
            R.write_subject_page(env, result["dir"], {
                "subject": subject, "title": title, "m": result["metrics"], "flags": result["flags"],
                "figures": result["figures"], "metadata": result["metadata"], "inputs": result["inputs"],
                "generated_at": generated_at, "app_version": app_version, "tool_version": nilearn.__version__,
            })
            seconds = round(time.monotonic() - t0, 2)
            outputs = sorted(_rel(p, output_dir) for p in result["dir"].iterdir() if p.is_file())
            provenance_subjects.append({
                "subject": subject, "inputs": sorted(_rel(p, input_dir) for p in inputs.files()),
                "outputs": outputs, "command": [sys.executable, "-m", "processor.main"], "seconds": seconds,
            })
            result["row"] = row
            results.append(result)
            rows.append(row)
            log.info("%s: ICV %s mL (%s), volumes from %s, %d flag(s)%s, %.1fs", subject,
                     f"{row['icv_mm3'] / 1000:.0f}" if row["icv_mm3"] else "?", row["icv_source"], row["volumes_source"],
                     len(result["flags"]), (": " + row["flags"]) if row["flags"] else "", seconds)
        except Exception as exc:  # noqa: BLE001 - one subject's failure must not sink the cohort
            seconds = round(time.monotonic() - t0, 2)
            log.error("%s: FAILED: %s\n%s", subject, exc, traceback.format_exc())
            skipped.append({"subject": subject, "reason": f"{type(exc).__name__}: {exc}"})
            provenance_subjects.append({
                "subject": subject, "inputs": sorted(_rel(p, input_dir) for p in inputs.files()), "outputs": [],
                "command": [sys.executable, "-m", "processor.main"], "seconds": seconds, "error": str(exc),
            })

    summary = M.cohort_summary(rows)
    flagged = {r["subject"] for r in results if r["flags"]}
    cohort_figs = figures.cohort(rows, flagged, report_dir) if rows else {}
    R.write_index(env, report_dir, {
        "title": title, "subjects": results, "skipped": skipped, "figures": cohort_figs, "summary": summary,
        "thresholds": thresholds, "generated_at": generated_at, "app_version": app_version,
        "tool_version": nilearn.__version__,
    })
    R.write_tables(tables_dir, rows, {
        "app": APP_NAME, "appVersion": app_version, "generatedAt": generated_at, "title": title,
        "thresholds": thresholds,
        "cohort": summary,
        "flags": {r["subject"]: r["flags"] for r in results},
        "subjects": {r["subject"]: {"metrics": r["metrics"], "flags": r["flags"], "inputs": r["inputs"],
                                    "figures": {k: f"report/{r['subject']}/{v}" for k, v in r["figures"].items() if v},
                                    "metadata": r["metadata"]} for r in results},
        "skipped": skipped,
    })

    finished = datetime.now(timezone.utc)
    provenance = {
        "app": APP_NAME, "appVersion": app_version, "tool": "nilearn", "toolVersion": nilearn.__version__,
        "startedAt": started.isoformat(timespec="seconds"), "finishedAt": finished.isoformat(timespec="seconds"),
        "parameters": parameters, "runtime": "lambda" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "container",
        "inputsFound": {name: [_rel(p, input_dir) for p in files] for name, files in found.items()},
        "subjects": provenance_subjects,
    }
    with open(report_dir / "provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    log.info("%d of %d subject(s) reported; wrote %s and %s", len(results), len(reportable), report_dir / "index.html",
             tables_dir / "volumes.csv")
    if not results:
        log.error("every subject failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
