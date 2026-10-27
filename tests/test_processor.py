"""End-to-end: the synthetic fixture through the processor, in both input layouts, plus the Lambda handler."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from processor import main as app
from processor import handler, inputs, metrics
from tests.make_fixture import make_fixture

SUBJECTS = ("sub-01", "sub-02")
EXPECTED_COLUMNS = {
    "subject", "icv_mm3", "icv_source", "left_hippocampus_mm3", "right_hippocampus_mm3", "brain_stem_mm3",
    "gm_pct_icv", "wm_pct_icv", "csf_pct_icv", "lateral_ventricles_pct_icv", "hippocampus_pct_icv",
    "amygdala_pct_icv", "thalamus_pct_icv", "caudate_pct_icv", "putamen_pct_icv", "pallidum_pct_icv",
    "cerebellum_pct_icv", "ai_hippocampus", "ai_amygdala", "ai_thalamus", "ai_caudate", "ai_putamen",
    "ai_pallidum", "ai_lateral_ventricle", "wmh_volume_mm3", "wmh_count", "wmh_pct_wm", "registration_dice",
    "synthseg_qc_min", "qc_hippocampus_amygdala", "flags",
}


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> dict[str, Path]:
    return make_fixture(tmp_path_factory.mktemp("fixture"), "both")


def run(input_dir: Path, output_dir: Path, monkeypatch, **params) -> int:
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for name in app.PARAMETERS:
        monkeypatch.delenv(name, raising=False)
    for name, value in params.items():
        monkeypatch.setenv(name, value)
    return app.main()


def read_rows(output_dir: Path) -> dict[str, dict[str, str]]:
    with open(output_dir / "tables" / "volumes.csv", newline="") as fh:
        return {r["subject"]: r for r in csv.DictReader(fh)}


def referenced_images(page: Path) -> list[Path]:
    return [page.parent / src for src in re.findall(r'<img src="([^"]+)"', page.read_text())]


@pytest.mark.parametrize("layout", ["linked", "flat"])
def test_both_layouts(fixture_root, tmp_path, monkeypatch, layout):
    out = tmp_path / "out"
    assert run(fixture_root[layout], out, monkeypatch) == 0

    rows = read_rows(out)
    assert set(rows) == set(SUBJECTS)
    assert EXPECTED_COLUMNS <= set(rows["sub-01"])
    for sub in SUBJECTS:
        assert rows[sub]["icv_source"] == "brain_mask"
        assert rows[sub]["volumes_source"] == "volumes_csv"
        assert 800 < float(rows[sub]["icv_mm3"]) / 1000 < 1000
        assert 30 < float(rows[sub]["gm_pct_icv"]) < 50
        assert 50 < float(rows[sub]["wm_pct_icv"]) < 70
        assert float(rows[sub]["qc_hippocampus_amygdala"]) > 0
    assert abs(float(rows["sub-01"]["ai_hippocampus"])) < 0.01
    assert float(rows["sub-02"]["ai_hippocampus"]) > 0.15
    assert rows["sub-01"]["wmh_count"] == "3" and float(rows["sub-01"]["wmh_volume_mm3"]) > 0
    assert rows["sub-02"]["wmh_count"] == "" and rows["sub-02"]["wmh_volume_mm3"] == ""
    assert float(rows["sub-01"]["registration_dice"]) == 0.93
    assert float(rows["sub-02"]["synthseg_qc_min"]) == 0.55

    m = json.loads((out / "tables" / "metrics.json").read_text())
    assert m["cohort"]["n"] == 2
    assert m["cohort"]["columns"]["gm_pct_icv"]["sd"] is not None
    assert m["subjects"]["sub-01"]["metrics"]["structures"]["left_hippocampus"] > 0
    flags = {sub: {f["check"] for f in m["flags"][sub]} for sub in SUBJECTS}
    # The synthetic head is small (~0.9 L), so ICV is flagged for both; sub-02 also has a small right
    # hippocampus, a poor registration and a low QC region.
    assert flags["sub-01"] == {"icv_low"}
    assert {"icv_low", "ai_hippocampus", "hippocampus_total", "registration_dice",
            "synthseg_qc_hippocampus_amygdala"} == flags["sub-02"]
    assert m["thresholds"]["icv_min_ml"] == 1000

    index = out / "report" / "index.html"
    assert index.exists()
    html = index.read_text()
    for sub in SUBJECTS:
        assert f'href="{sub}/report.html"' in html
        page = out / "report" / sub / "report.html"
        assert page.exists()
        images = referenced_images(page)
        assert images, "subject page references no figures"
        for img in images:
            assert img.exists(), img
        names = {p.name for p in images}
        assert {"segmentation.png", "volumes.png", "mni.png"} <= names
    assert "wmh.png" in {p.name for p in referenced_images(out / "report" / "sub-01" / "report.html")}
    assert "wmh.png" not in {p.name for p in referenced_images(out / "report" / "sub-02" / "report.html")}
    for img in referenced_images(index):
        assert img.exists(), img
    assert "<script" not in html and "http" not in re.sub(r"https?://schemas", "", html).split("<style>")[1].split("</style>")[0]
    assert "Siemens" in (out / "report" / "sub-01" / "report.html").read_text()

    prov = json.loads((out / "report" / "provenance.json").read_text())
    assert prov["app"] == "processor-brain-volumetrics"
    assert prov["tool"] == "nilearn" and prov["toolVersion"]
    assert {s["subject"] for s in prov["subjects"]} == set(SUBJECTS)
    first = prov["subjects"][0]
    assert first["inputs"] and first["outputs"] and first["seconds"] >= 0
    assert all(not os.path.isabs(p) for p in first["inputs"] + first["outputs"])


def test_threshold_override_and_title(fixture_root, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(fixture_root["linked"], out, monkeypatch, REPORT_TITLE="My Cohort",
               FLAG_THRESHOLDS='{"icv_min_ml": 700, "ai_hippocampus_max": 0.9}') == 0
    m = json.loads((out / "tables" / "metrics.json").read_text())
    assert m["flags"]["sub-01"] == []
    assert {f["check"] for f in m["flags"]["sub-02"]} == {"hippocampus_total", "registration_dice",
                                                        "synthseg_qc_hippocampus_amygdala"}
    assert m["thresholds"]["icv_min_ml"] == 700 and m["thresholds"]["icv_max_ml"] == 2000
    assert "My Cohort" in (out / "report" / "index.html").read_text()


def test_bad_thresholds(fixture_root, tmp_path, monkeypatch):
    assert run(fixture_root["linked"], tmp_path / "out", monkeypatch, FLAG_THRESHOLDS="[1]") == 2


def test_labels_only(fixture_root, tmp_path, monkeypatch):
    """Only label maps: volumes are counted from them, ICV from the labelled voxels, no figures but the bars."""
    src = fixture_root["linked"]
    in_dir = tmp_path / "in"
    shutil.copytree(src / "labels", in_dir / "labels")
    out = tmp_path / "out"
    assert run(in_dir, out, monkeypatch) == 0
    rows = read_rows(out)
    assert set(rows) == set(SUBJECTS)
    assert rows["sub-01"]["volumes_source"] == "labels"
    assert rows["sub-01"]["icv_source"] == "labelled_voxels"
    # Counted from the label map and read from the SynthSeg CSV (which the fixture made from the same map) agree.
    full = tmp_path / "full"
    assert run(src, full, monkeypatch) == 0
    for key in ("left_hippocampus_mm3", "right_cerebral_white_matter_mm3", "brain_stem_mm3"):
        assert float(rows["sub-01"][key]) == pytest.approx(float(read_rows(full)["sub-01"][key]))
    assert rows["sub-01"]["registration_dice"] == "" and rows["sub-01"]["synthseg_qc_min"] == ""
    page = out / "report" / "sub-01" / "report.html"
    assert {p.name for p in referenced_images(page)} == {"segmentation.png", "volumes.png"}


def test_volumes_only_flat(fixture_root, tmp_path, monkeypatch):
    """A lone combined volumes.csv at the root (two subjects, 'subject' header) is enough for tables and pages."""
    src = fixture_root["linked"]
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    header = None
    rows_out = []
    for sub in SUBJECTS:
        with open(src / "volumes" / f"{sub}_volumes.csv", newline="") as fh:
            h, row = list(csv.reader(fh))
        header = ["subject"] + h[1:]
        rows_out.append([sub] + row[1:])
    with open(in_dir / "volumes.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows_out)
    out = tmp_path / "out"
    assert run(in_dir, out, monkeypatch) == 0
    rows = read_rows(out)
    assert set(rows) == set(SUBJECTS)
    assert rows["sub-01"]["icv_source"] == "synthseg_total_intracranial"
    assert float(rows["sub-01"]["hippocampus_pct_icv"]) > 0
    assert (out / "report" / "sub-02" / "report.html").exists()


def test_no_inputs(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "notes.txt").write_text("nothing useful")
    assert run(in_dir, tmp_path / "out", monkeypatch) == 1
    assert (tmp_path / "out" / "report").is_dir() and (tmp_path / "out" / "tables").is_dir()


def test_read_synthseg_csv_variants(tmp_path):
    p = tmp_path / "a_volumes.csv"
    p.write_text("subject,total intracranial,Left-Hippocampus,right hippocampus,unknown thing\n"
                 "/x/sub-07_T1w.nii.gz,1500000,4000,3900,1\nsub-08,1400000,3800,3700,2\n")
    rows = inputs.read_synthseg_csv(p)
    assert set(rows) == {"sub-07", "sub-08"}
    known, extra = metrics.structures_from_row(rows["sub-07"])
    assert known == {"left_hippocampus": 4000.0, "right_hippocampus": 3900.0}
    assert extra == {"total_intracranial": 1500000.0, "unknown_thing": 1.0}
    q = tmp_path / "sub-09_qc.csv"
    q.write_text(",general white matter,hippocampus+amygdala\n/in/sub-09.nii.gz,0.9,0.6\n")
    assert inputs.read_synthseg_csv(q) == {"sub-09": {"general_white_matter": "0.9", "hippocampus_amygdala": "0.6"}}
    n = tmp_path / "sub-10_volumes.csv"
    n.write_text("left hippocampus,right hippocampus\n10,20\n")
    assert inputs.read_synthseg_csv(n) == {"sub-10": {"left_hippocampus": "10", "right_hippocampus": "20"}}


def test_handler(fixture_root, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("REPORT_TITLE", "stale value from a warm function")
    event = {"inputDir": str(fixture_root["linked"]), "outputDir": str(out),
             "FLAG_THRESHOLDS": {"icv_min_ml": 700}}
    result = handler.handler(event, context=None)
    assert result == {"status": "ok", "outputDir": str(out)}
    assert "REPORT_TITLE" not in os.environ
    assert os.environ["FLAG_THRESHOLDS"] == '{"icv_min_ml": 700}'
    m = json.loads((out / "tables" / "metrics.json").read_text())
    assert m["title"] == "Clinical Brain MRI Volumetrics" and m["flags"]["sub-01"] == []
    assert (out / "report" / "index.html").exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        handler.handler({"inputDir": str(empty), "outputDir": str(tmp_path / "out2")}, context=None)
