"""
The HTML pages (Jinja2, inline CSS, no scripts: they work offline from a
dataset folder) and the cohort tables.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from processor.metrics import ASYMMETRY_PAIRS, PAIRS, PCT_PAIRS, STRUCTURE_KEYS

TEMPLATES = Path(__file__).parent / "templates"

SCANNER_FIELDS = [
    "Manufacturer", "ManufacturersModelName", "ManufacturerModelName", "MagneticFieldStrength", "StationName",
    "SoftwareVersions", "ScanningSequence", "SequenceName", "ProtocolName", "RepetitionTime", "EchoTime",
    "InversionTime", "FlipAngle", "scanner", "manufacturer", "model", "field_strength", "fieldStrength",
]


def _num(value: Any, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value):{'+' if signed else ''}.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _ml(value: Any, digits: int = 0) -> str:
    return "–" if value is None else _num(float(value) / 1000.0, digits)


def _label(key: str) -> str:
    return str(key).replace("_", " ")


def environment() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    env.filters["num"] = _num
    env.filters["ml"] = _ml
    env.filters["label"] = _label
    return env


def _join(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " × ".join(_num(v, 2) if isinstance(v, (int, float)) else str(v) for v in value)
    return str(value)


def header_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """Modalities, voxel size, shape and scanner fields from the images.json entry and the sidecars."""
    image = metadata.get("image") or {}
    images = [image] + list(metadata.get("other_images") or [])
    sidecars = metadata.get("sidecars") or {}
    modalities = []
    for entry in images:
        mod = entry.get("modality") or entry.get("suffix") or entry.get("type")
        if mod and mod not in modalities:
            modalities.append(str(mod))
    for name in sidecars:
        mod = name.rsplit(".", 1)[0].split("_")[-1]
        if mod not in modalities:
            modalities.append(mod)
    voxel = next((image.get(k) for k in ("voxelSize", "voxel_size", "voxelSizeMm", "pixdim", "zooms", "resolution") if image.get(k)), None)
    shape = next((image.get(k) for k in ("shape", "dimensions", "dim", "size") if image.get(k)), None)
    scanner: dict[str, Any] = {}
    sources = images + list(sidecars.values())
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in SCANNER_FIELDS:
            if key in source and key not in scanner and source[key] not in (None, ""):
                scanner[key] = source[key]
    return {
        "modalities": ", ".join(modalities),
        "voxel_size": (_join(voxel) + " mm") if voxel else "",
        "shape": _join(shape) if shape else "",
        "scanner": scanner,
    }


def structure_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    s = metrics["structures"]
    rows = []
    for pair, (left, right) in PAIRS.items():
        if s.get(left) is None and s.get(right) is None:
            continue
        rows.append({
            "name": _label(pair), "left": s.get(left), "right": s.get(right), "total": metrics["totals"].get(pair),
            "pct": metrics["pct_icv"].get(pair) if pair in PCT_PAIRS else
                   (metrics["pct_icv"].get("lateral_ventricles") if pair == "lateral_ventricle" else None),
            "ai": metrics["asymmetry"].get(pair) if pair in ASYMMETRY_PAIRS else None,
        })
    for key in ("brain_stem", "third_ventricle", "fourth_ventricle", "csf"):
        if s.get(key) is not None:
            icv = metrics["icv"]["mm3"]
            rows.append({"name": _label(key), "left": None, "right": None, "total": s[key],
                         "pct": 100.0 * s[key] / icv if icv else None, "ai": None})
    return rows


def write_subject_page(env: Environment, out_dir: Path, context: dict[str, Any]) -> Path:
    page = out_dir / "report.html"
    context = dict(context)
    context["structure_rows"] = structure_rows(context["m"])
    context["header"] = header_fields(context.get("metadata") or {})
    sidecars = (context.get("metadata") or {}).get("sidecars") or {}
    context["sidecars"] = {name: {k: v for k, v in fields.items() if not isinstance(v, (dict, list))}
                           for name, fields in sidecars.items() if isinstance(fields, dict)}
    page.write_text(env.get_template("subject.html.j2").render(**context), encoding="utf-8")
    return page


def write_index(env: Environment, out_dir: Path, context: dict[str, Any]) -> Path:
    page = out_dir / "index.html"
    page.write_text(env.get_template("index.html.j2").render(**context), encoding="utf-8")
    return page


def write_tables(tables_dir: Path, rows: list[dict[str, Any]], metrics_json: dict[str, Any]) -> list[Path]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    csv_path = tables_dir / "volumes.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    json_path = tables_dir / "metrics.json"
    with open(json_path, "w") as fh:
        json.dump(metrics_json, fh, indent=2)
    return [csv_path, json_path]


__all__ = ["environment", "write_subject_page", "write_index", "write_tables", "header_fields", "STRUCTURE_KEYS"]
