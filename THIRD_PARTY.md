# Third-party software

This app wraps no external CLI; it is Python on top of these libraries, all installed from PyPI at the versions pinned in `requirements.txt`.

| Component | Version | Licence | Use | Link |
|---|---|---|---|---|
| nilearn | 0.11.1 | BSD-3-Clause | Overlay figures (`plot_roi`, `plot_anat`, template edges) and the bundled MNI152 template | https://nilearn.github.io |
| nibabel | 5.3.2 | MIT | Reading NIfTI label maps, masks and images | https://nipy.org/nibabel |
| NumPy | 2.2.6 | BSD-3-Clause | Voxel counting, cohort statistics | https://numpy.org |
| pandas | 2.2.3 | BSD-3-Clause | Reading the SynthSeg / lesion CSVs | https://pandas.pydata.org |
| Matplotlib | 3.10.3 | PSF-based (Matplotlib licence) | Bar charts; Agg backend | https://matplotlib.org |
| Jinja2 | 3.1.6 | BSD-3-Clause | HTML templates | https://jinja.palletsprojects.com |
| SciPy, scikit-learn, joblib, lxml, requests, packaging | as resolved | BSD-3-Clause / Apache-2.0 | nilearn's dependencies; `scipy.ndimage.label` counts WMH components | |
| awslambdaric | 2.2.1 | Apache-2.0 | Lambda runtime interface client (Lambda mode only) | https://github.com/aws/aws-lambda-python-runtime-interface-client |
| python:3.12-slim | Docker Official Image | PSF (Python) on Debian (various free licences) | Base image | https://hub.docker.com/_/python |

The MNI152 template nilearn ships (`load_mni152_template`) is derived from the ICBM 152 nonlinear atlases, © Copyright 1993–2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University; free for use under its own licence (see nilearn's documentation).

## Citations

- nilearn: Abraham A, Pedregosa F, Eickenberg M, Gervais P, Mueller A, Kossaifi J, Gramfort A, Thirion B, Varoquaux G. *Machine learning for neuroimaging with scikit-learn.* Frontiers in Neuroinformatics 2014; 8:14. https://doi.org/10.3389/fninf.2014.00014
- SynthSeg (the label scheme, volumes and QC the report reads): Billot B, Greve DN, Puonti O, Thielscher A, Van Leemput K, Fischl B, Dalca AV, Iglesias JE. *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining.* Medical Image Analysis 2023; 86:102789. And Billot B, Magdamo C, Cheng Y, Arnold SE, Das S, Iglesias JE. *Robust machine learning segmentation for large-scale analytical multi-site brain MRI studies.* PNAS 2023; 120(9):e2216399120 (QC, robust version).
- The label values follow FreeSurfer's `FreeSurferColorLUT.txt` (Fischl B. *FreeSurfer.* NeuroImage 2012; 62(2):774–781).
