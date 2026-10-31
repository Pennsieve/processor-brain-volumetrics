# sample-input

This app consumes the *outputs* of the other Clinical Brain MRI steps (synthseg, synthstrip,
ants-mni, lst-ai, nifti-prep), so there is no public dataset to download for it directly.

For a local run, generate the synthetic two-subject fixture instead:

```sh
make fixture      # writes data/fixture/linked/ and data/fixture/flat/ (gitignored)
make run          # runs the Docker image over data/fixture/linked/
```

`tests/make_fixture.py` builds a 64³, 2 mm ellipsoid "brain" per subject with blobs carrying the
FreeSurfer/SynthSeg labels, a brain mask, a SynthSeg-style `<sub>_volumes.csv` and `<sub>_qc.csv`, a
WMH mask and `<sub>_lesions.csv` (sub-01 only), a `<sub>_registration.json` and `images.json` — in
the linked layout (`INPUT_DIR/<input>/…`) and in a flat, nested layout.

To try real data, run the upstream steps on an OpenNeuro T1w, e.g.

```sh
aws s3 cp --no-sign-request s3://openneuro.org/ds000117/sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz data/sub-01_T1w.nii.gz
```

and point `INPUT_DIR` at a folder holding their outputs (linked, or all together).
