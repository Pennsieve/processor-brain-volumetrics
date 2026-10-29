# processor-brain-volumetrics

The final step of the **Clinical Brain MRI Volumetrics** workflow, and the only one whose output is
for a person. It gathers what the upstream steps produced, works out per-subject volumetrics, and
writes an HTML report (cohort page + one page per subject, with figures) and cohort tables.

It degrades gracefully: **every input is optional**. A subject gets a report when it has a SynthSeg
volumes CSV *or* a label map; everything else (QC, mask, brain image, MNI-warped image, registration
check, lesions, metadata) adds rows, flags and figures when present and is simply absent otherwise.
Files from different inputs are matched by their BIDS subject label (everything before the first `_`).

## What it computes

Per subject (all volumes in mm³; the pages show mL):

- **Structure volumes** for the SynthSeg/FreeSurfer label set (cerebral WM and cortex, lateral and
  inferior lateral ventricles, cerebellum WM and cortex, thalamus, caudate, putamen, pallidum,
  hippocampus, amygdala, accumbens, ventral DC, 3rd/4th ventricle, brain stem, CSF), read from the
  volumes CSV (columns matched case-insensitively, unknown ones ignored) or, when there is none,
  counted from the label map (voxels × voxel volume).
- **ICV** from the brain mask when present, else SynthSeg's `total intracranial`, else the labelled
  volume (a brain-volume stand-in, named as such in `icv_source`).
- **Tissue totals and % of ICV**: GM (cerebral + cerebellar cortex + subcortical GM), WM (cerebral +
  cerebellar WM), CSF (ventricles + CSF label), lateral ventricles, hippocampus, amygdala, thalamus,
  caudate, putamen, pallidum, cerebellum. Brain stem is reported on its own, in neither GM nor WM.
- **Asymmetry index** `ai_<structure> = 2·(L − R)/(L + R)` for hippocampus, amygdala, thalamus,
  caudate, putamen, pallidum and lateral ventricle.
- **WMH** volume, count and % of WM from `<sub>_lesions.csv`, else from the WMH mask (connected
  components).
- **Registration Dice** from `<sub>_registration.json`; **SynthSeg QC** per region and its minimum.
- **Flags**: the checks below that fall outside their plausible range.

## Flag thresholds

These are **screening heuristics, not clinical cut-offs**. They mark a subject for a second look;
they do not diagnose. Override any of them with `FLAG_THRESHOLDS`.

| Key | Default | Flag when |
|---|---|---|
| `icv_min_ml` / `icv_max_ml` | 1000 / 2000 | ICV outside 1000–2000 mL |
| `ai_hippocampus_max` | 0.15 | \|hippocampal asymmetry index\| > 0.15 |
| `lateral_ventricles_pct_icv_max` | 5 | lateral ventricles (L+R) > 5 % of ICV |
| `hippocampus_total_min_ml` | 5 | hippocampus (L+R) < 5 mL |
| `registration_dice_min` | 0.85 | MNI registration Dice < 0.85 |
| `synthseg_qc_min` | 0.65 | any SynthSeg QC region < 0.65 |

A check whose inputs are missing is not evaluated (no mask and no volumes → no ICV flag, etc.).

## Ports

Inputs (all `placement: subfolder`, none required). Linked from a workflow, each arrives in
`INPUT_DIR/<input>/`; without links (or on an older compute node) everything is at the root of
`INPUT_DIR`, possibly nested, and the app finds files by their BIDS names (`bids_names.py`).

| Input | Files | Produced by | Used for |
|---|---|---|---|
| `volumes` | `<sub>_volumes.csv` (SynthSeg `--vol`, one row) and/or a combined `volumes.csv` | synthseg | structure volumes, `total intracranial` |
| `qc` | `<sub>_qc.csv` (SynthSeg `--qc`) | synthseg | QC scores, QC flags |
| `labels` | `<sub>_dseg.nii.gz` (FreeSurfer label values) | synthseg | segmentation figure; volumes when `volumes` is missing |
| `brain` | `<sub>_desc-brain_T1w.nii.gz` | synthstrip | background of the overlay figures |
| `mask` | `<sub>_desc-brain_mask.nii.gz` | synthstrip | ICV |
| `warped` | `<sub>_space-MNI152NLin2009cAsym_T1w.nii.gz` | ants-mni | MNI figure with template edges |
| `transforms` | `<sub>_registration.json` `{dice, …}` (the `.mat`/warps are ignored) | ants-mni | registration check |
| `lesions` | `<sub>_desc-wmh_mask.nii.gz` (+ `_desc-wmh_probseg.nii.gz`) | lst-ai | WMH figure; WMH stats when no lesion table |
| `lesion-volumes` | `<sub>_lesions.csv` (`subject,lesion_volume_mm3,lesion_count`) | lst-ai | WMH volume/count |
| `metadata` | `images.json` (list of `{subject, modality, voxelSize, shape, scanner fields…}`) and `<sub>_T1w.json` sidecars | nifti-prep | page header |

Outputs (two, so each goes to `OUTPUT_DIR/<output>/`):

| Output | Files |
|---|---|
| `report` | `index.html` (cohort table with key metrics, flags and links; cohort bar charts; cohort summary; thresholds), `<sub>/report.html` + `segmentation.png`, `volumes.png`, `mni.png`, `wmh.png` (each when its inputs exist), `provenance.json` |
| `tables` | `volumes.csv` (one row per subject), `metrics.json` (the same nested, plus cohort n/mean/sd per derived column and the flags) |

The pages are self-contained: inline CSS, no scripts, PNGs referenced relatively, so they open
offline straight from a dataset folder.

`volumes.csv` columns: `subject`, `icv_mm3`, `icv_source`, `volumes_source`, `<structure>_mm3` for
every label, `<tissue|pair>_total_mm3`, `gm_pct_icv`, `wm_pct_icv`, `csf_pct_icv`,
`lateral_ventricles_pct_icv`, `<structure>_pct_icv`, `cerebellum_pct_icv`, `ai_<structure>`,
`wmh_volume_mm3`, `wmh_count`, `wmh_pct_wm`, `registration_dice`, `synthseg_qc_min`, `qc_<region>`,
`n_flags`, `flags` (`;`-separated check names). Missing values are empty.

## Parameters

| Parameter | Type | Default | |
|---|---|---|---|
| `REPORT_TITLE` | string | `Clinical Brain MRI Volumetrics` | Title on the cohort and subject pages |
| `FLAG_THRESHOLDS` | string | `{}` | JSON object overriding the thresholds above, e.g. `{"icv_min_ml": 900, "synthseg_qc_min": 0.7}` |

## Runtimes

`runtime`: cpu 2048, memory 8192, `timeoutSeconds` 900, `computeTypes: [standard, lambda]`. The image
runs either way:

- **standard** (the default): an ECS task. `INPUT_DIR`, `OUTPUT_DIR` and the parameters are
  environment variables, and `main()` runs once.
- **lambda**: a workflow step set to run as a Lambda function. Lambda sets `AWS_LAMBDA_RUNTIME_API`,
  so `processor/entry.py` hands over to the Lambda runtime interface client, and
  `processor/handler.py` takes `inputDir`, `outputDir` and the parameters from the payload and runs
  the same `main()`. A failed run raises, so the step fails.

A subject that fails is logged and skipped (and listed on the cohort page); the run exits 1 only if
every subject failed, or if no subject had a volumes CSV or label map at all. Matplotlib uses the Agg
backend and `MPLCONFIGDIR=/tmp/matplotlib`, so the image runs as uid 1000 without a writable home.

## Running it locally

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python     # pytest: the synthetic fixture through both layouts, the Lambda handler
make fixture PYTHON=.venv/bin/python  # data/fixture/{linked,flat}/ (gitignored)
make run PYTHON=.venv/bin/python      # Docker, over data/fixture/linked/, into data/output/
open data/output/report/index.html
```

`tests/make_fixture.py` makes two synthetic subjects (64³, 2 mm): an ellipsoid "brain" with blobs
carrying the FreeSurfer labels, a mask, a SynthSeg-style volumes and QC CSV, a WMH mask and lesion
table for sub-01, a registration.json and images.json. The head is small (ICV ≈ 0.9 L), so the ICV
flag fires on both; sub-02 also has a small right hippocampus, a Dice of 0.80 and a QC region at 0.55.

Over any real folder:

```sh
INPUT_DIR=/path/to/inputs OUTPUT_DIR=/path/to/out .venv/bin/python -m processor.main
```

As a Lambda function, with the [runtime interface emulator](https://github.com/aws/aws-lambda-runtime-interface-emulator):

```sh
docker build -t processor-brain-volumetrics .
docker run --rm -p 9000:8080 -v "$PWD/data/fixture/linked:/in:ro" -v "$PWD/aws-lambda-rie:/rie:ro" \
  --entrypoint /rie processor-brain-volumetrics python3 -m processor.entry
curl -XPOST localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"inputDir": "/in", "outputDir": "/tmp/out", "FLAG_THRESHOLDS": "{}"}'
```

## Provenance

`report/provenance.json`: `app`, `appVersion` (from `VERSION`), `tool`/`toolVersion` (nilearn),
`startedAt`/`finishedAt`, `parameters`, `inputsFound` per input, and `subjects` with each subject's
input and output paths (relative), the command and the seconds taken.

## Citations

- nilearn: Abraham A, Pedregosa F, Eickenberg M, Gervais P, Mueller A, Kossaifi J, Gramfort A,
  Thirion B, Varoquaux G. Machine learning for neuroimaging with scikit-learn. *Front Neuroinform*
  2014; 8:14. https://doi.org/10.3389/fninf.2014.00014
- SynthSeg (label scheme, volumes and QC): Billot B, Greve DN, Puonti O, Thielscher A, Van Leemput K,
  Fischl B, Dalca AV, Iglesias JE. SynthSeg: Segmentation of brain MRI scans of any contrast and
  resolution without retraining. *Med Image Anal* 2023; 86:102789; and Billot B et al. Robust machine
  learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets. *PNAS* 2023;
  120(9):e2216399120.

See `THIRD_PARTY.md` for licences.
