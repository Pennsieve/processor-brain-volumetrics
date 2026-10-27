"""
Finding each input's files and grouping them by subject.

Linked layout: INPUT_DIR/<input name>/ (placement: subfolder). Flat layout
(no links, or an older compute node): everything at the root of INPUT_DIR,
possibly nested, picked by BIDS suffix and entities (see bids_names.py).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from processor import bids_names
from processor.metrics import normalise_column, to_float

log = logging.getLogger("processor-brain-volumetrics")

INPUT_NAMES = ["volumes", "qc", "labels", "brain", "mask", "warped", "transforms", "lesions", "lesion-volumes", "metadata"]


@dataclass
class SubjectInputs:
    subject: str
    volumes_row: dict[str, Any] | None = None       # normalised SynthSeg --vol row
    volumes_file: Path | None = None
    qc_row: dict[str, Any] | None = None            # normalised SynthSeg --qc row
    qc_file: Path | None = None
    labels: Path | None = None
    brain: Path | None = None
    mask: Path | None = None
    warped: Path | None = None
    registration: Path | None = None
    lesion_mask: Path | None = None
    lesion_probseg: Path | None = None
    lesion_row: dict[str, Any] | None = None        # {lesion_volume_mm3, lesion_count}
    lesion_file: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)   # images.json entry + sidecar fields
    metadata_files: list[Path] = field(default_factory=list)

    def files(self) -> list[Path]:
        paths = [self.volumes_file, self.qc_file, self.labels, self.brain, self.mask, self.warped,
                 self.registration, self.lesion_mask, self.lesion_probseg, self.lesion_file]
        return [p for p in paths if p is not None] + list(self.metadata_files)

    @property
    def reportable(self) -> bool:
        return self.volumes_row is not None or self.labels is not None


def _pick(input_dir: Path, name: str, suffixes: Iterable[str], exts: Iterable[str],
          prefer: Callable[[Path], bool] | None = None) -> list[Path]:
    """find_input_files, then keep the files `prefer` accepts. When the input's own folder exists and the
    filter leaves nothing, every file in that folder counts (it was linked there on purpose); without the
    folder only preferred files count, so a root search can't mistake e.g. the MNI-warped dseg for labels."""
    files = bids_names.find_input_files(input_dir, name, suffixes, exts)
    if prefer is None:
        return files
    preferred = [f for f in files if prefer(f)]
    if preferred or not (input_dir / name).is_dir():
        return preferred
    return files


def _subject_from_cell(value: Any, fallback: str) -> str:
    """The subject label from a SynthSeg CSV subject cell (a label, a file name or a path), else the file's."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    return bids_names.subject_label(Path(text).name)


def read_synthseg_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Rows of a SynthSeg --vol or --qc CSV keyed by subject label, with normalised column names. SynthSeg
    writes 'subject' as the first header for a combined file and an empty header for a one-subject file;
    both are accepted, as is a file without any subject column (the file name then names the subject)."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        return {}
    columns = [normalise_column(c) for c in frame.columns]
    first = frame.columns[0]
    has_subject_col = columns[0] in ("", "subject") or to_float(frame[first].iloc[0]) is None
    rows: dict[str, dict[str, Any]] = {}
    for _, raw in frame.iterrows():
        row: dict[str, Any] = {}
        for col, norm in zip(frame.columns, columns):
            if col == first and has_subject_col:
                continue
            if norm:
                row[norm] = raw[col]
        subject = _subject_from_cell(raw[first], bids_names.subject_label(path)) if has_subject_col \
            else bids_names.subject_label(path)
        rows.setdefault(subject, row)
    return rows


def read_lesion_csv(path: Path) -> dict[str, dict[str, Any]]:
    """`subject,lesion_volume_mm3,lesion_count` rows keyed by subject label (tolerant of column case)."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows: dict[str, dict[str, Any]] = {}
    columns = {normalise_column(c): c for c in frame.columns}
    for _, raw in frame.iterrows():
        subject = _subject_from_cell(raw[columns["subject"]], bids_names.subject_label(path)) \
            if "subject" in columns else bids_names.subject_label(path)
        volume = next((to_float(raw[columns[c]]) for c in ("lesion_volume_mm3", "wmh_volume_mm3", "volume_mm3") if c in columns), None)
        count = next((to_float(raw[columns[c]]) for c in ("lesion_count", "wmh_count", "count") if c in columns), None)
        rows.setdefault(subject, {"volume_mm3": volume, "count": int(count) if count is not None else None})
    return rows


def _load_json(path: Path) -> Any:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", path, exc)
        return None


def _subject_of_entry(entry: dict[str, Any]) -> str | None:
    for key in ("subject", "participant", "sub", "file", "filename", "path", "name"):
        if entry.get(key):
            return bids_names.subject_label(Path(str(entry[key])).name)
    return None


def gather(input_dir: Path) -> tuple[dict[str, SubjectInputs], dict[str, list[Path]]]:
    """Every subject's inputs, and the files found per input (for logging and the provenance)."""
    found: dict[str, list[Path]] = {}
    subjects: dict[str, SubjectInputs] = {}

    def subj(label: str) -> SubjectInputs:
        return subjects.setdefault(label, SubjectInputs(subject=label))

    nifti = bids_names.NIFTI_EXTS
    no_space = lambda p: "space" not in bids_names.entities_of(p)  # noqa: E731
    desc_is = lambda value: (lambda p: bids_names.entities_of(p).get("desc") == value)  # noqa: E731

    found["volumes"] = _pick(input_dir, "volumes", ("volumes",), (".csv",))
    for path in found["volumes"]:
        for label, row in read_synthseg_csv(path).items():
            s = subj(label)
            if s.volumes_row is None:
                s.volumes_row, s.volumes_file = row, path

    found["qc"] = _pick(input_dir, "qc", ("qc",), (".csv",))
    for path in found["qc"]:
        for label, row in read_synthseg_csv(path).items():
            s = subj(label)
            if s.qc_row is None:
                s.qc_row, s.qc_file = row, path

    found["labels"] = _pick(input_dir, "labels", ("dseg",), nifti, prefer=no_space)
    for label, path in bids_names.by_subject(found["labels"]).items():
        subj(label).labels = path

    found["brain"] = _pick(input_dir, "brain", ("T1w",), nifti, prefer=desc_is("brain"))
    for label, path in bids_names.by_subject(found["brain"]).items():
        subj(label).brain = path

    found["mask"] = _pick(input_dir, "mask", ("mask",), nifti, prefer=desc_is("brain"))
    for label, path in bids_names.by_subject(found["mask"]).items():
        subj(label).mask = path

    found["warped"] = _pick(input_dir, "warped", ("T1w",), nifti,
                            prefer=lambda p: bids_names.entities_of(p).get("space", "").upper().startswith("MNI"))
    for label, path in bids_names.by_subject(found["warped"]).items():
        subj(label).warped = path

    found["transforms"] = _pick(input_dir, "transforms", ("registration",), (".json",))
    for label, path in bids_names.by_subject(found["transforms"]).items():
        subj(label).registration = path

    found["lesions"] = _pick(input_dir, "lesions", ("mask", "probseg"), nifti, prefer=desc_is("wmh"))
    for path in found["lesions"]:
        s = subj(bids_names.subject_label(path))
        if bids_names.suffix_of(path) == "mask" and s.lesion_mask is None:
            s.lesion_mask = path
        elif bids_names.suffix_of(path) == "probseg" and s.lesion_probseg is None:
            s.lesion_probseg = path

    found["lesion-volumes"] = _pick(input_dir, "lesion-volumes", ("lesions",), (".csv",))
    for path in found["lesion-volumes"]:
        for label, row in read_lesion_csv(path).items():
            s = subj(label)
            if s.lesion_row is None:
                s.lesion_row, s.lesion_file = row, path

    # metadata: images.json (a list of per-image records) and BIDS sidecars <sub>_T1w.json etc.
    found["metadata"] = _pick(input_dir, "metadata", ("images", "T1w", "FLAIR", "T2w"), (".json",))
    for path in found["metadata"]:
        content = _load_json(path)
        if content is None:
            continue
        if path.name == "images.json":
            entries = content.get("images", content) if isinstance(content, dict) else content
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                label = _subject_of_entry(entry)
                if label is None:
                    continue
                s = subj(label)
                if "image" not in s.metadata:
                    s.metadata["image"] = entry
                else:
                    s.metadata.setdefault("other_images", []).append(entry)
                if path not in s.metadata_files:
                    s.metadata_files.append(path)
        elif isinstance(content, dict):
            s = subj(bids_names.subject_label(path))
            s.metadata.setdefault("sidecars", {})[path.name] = content
            s.metadata_files.append(path)

    # Subjects that only turned up through side inputs (no volumes and no labels) can't get a report.
    return dict(sorted(subjects.items())), found
