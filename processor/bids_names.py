"""Shared naming helpers for the Clinical Brain MRI apps (copied verbatim into every repo)."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Iterable

NIFTI_EXTS = (".nii.gz", ".nii")


def strip_nifti_ext(name: str) -> str:
    for ext in NIFTI_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def subject_label(path: str | os.PathLike) -> str:
    """'sub-01_desc-brain_T1w.nii.gz' -> 'sub-01'; 'brain.nii.gz' -> 'brain'."""
    return strip_nifti_ext(Path(path).name).split("_")[0]


def suffix_of(path: str | os.PathLike) -> str:
    """BIDS suffix: the last underscore-separated token of the stem ('T1w', 'mask', 'dseg', ...)."""
    return strip_nifti_ext(Path(path).name).split("_")[-1]


def entities_of(path: str | os.PathLike) -> dict[str, str]:
    """{'desc': 'brain', 'space': 'MNI152NLin2009cAsym', ...} from the file name."""
    out = {}
    for tok in strip_nifti_ext(Path(path).name).split("_")[1:-1]:
        m = re.fullmatch(r"([a-zA-Z0-9]+)-([a-zA-Z0-9.]+)", tok)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def derived_name(source: str | os.PathLike, suffix: str, ext: str = ".nii.gz", **entities: str) -> str:
    """derived_name('sub-01_T1w.nii.gz', 'T1w', desc='brain') -> 'sub-01_desc-brain_T1w.nii.gz'.
    Entity order follows BIDS: space before desc; others keep insertion order."""
    sub = subject_label(source)
    order = ["space", "desc"]
    keys = [k for k in order if k in entities] + [k for k in entities if k not in order]
    parts = [sub] + [f"{k}-{entities[k]}" for k in keys] + [suffix]
    return "_".join(parts) + ext


def find_input_files(input_dir: str | os.PathLike, name: str, suffixes: Iterable[str], exts: Iterable[str] = NIFTI_EXTS) -> list[Path]:
    """Files for input `name`: INPUT_DIR/<name>/ if it exists (recursively), else all of INPUT_DIR
    recursively, keeping files whose BIDS suffix is in `suffixes` (case-insensitive) and whose
    extension is in `exts`. Sorted, de-duplicated by file name, symlinks allowed."""
    root = Path(input_dir)
    base = root / name
    if not base.is_dir():
        base = root
    wanted = {s.lower() for s in suffixes}
    exts = tuple(exts)
    seen: dict[str, Path] = {}
    for p in sorted(base.rglob("*")):
        if not p.is_file() or not p.name.endswith(exts):
            continue
        if wanted and suffix_of(p).lower() not in wanted:
            continue
        seen.setdefault(p.name, p)
    return list(seen.values())


def by_subject(files: Iterable[Path]) -> dict[str, Path]:
    """Map subject label -> file (first one wins per subject)."""
    out: dict[str, Path] = {}
    for f in files:
        out.setdefault(subject_label(f), f)
    return out


def output_dir_for(output_dir: str | os.PathLike, outputs: list[str], name: str) -> Path:
    """Where output `name` goes: OUTPUT_DIR itself for a single-output app, else OUTPUT_DIR/<name>/."""
    base = Path(output_dir)
    d = base if len(outputs) == 1 else base / name
    d.mkdir(parents=True, exist_ok=True)
    return d
