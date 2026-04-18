# Hoegn Index

![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)
![Python: 3.10](https://img.shields.io/badge/Python-3.10-brightgreen.svg)
![R: ≥4.2.3](https://img.shields.io/badge/R-%E2%89%A54.2.3-276DC3.svg)

> A concise, versatile, and conservative hyperparameter selection framework for dimensionality reduction embeddings. Given any set of quality metrics and any embedding algorithm, the Hoegn Index finds the parameter configuration with the lowest chance of catastrophic metric failure — making it especially suited for multimodal neuroscience data where ground truth is unavailable and parameter choices carry high interpretive stakes.

---

## Overview

![Pipeline overview](figures/pipeline_overview.png)

Dimensionality reduction algorithms such as UMAP, PHATE, and DIABLO are now standard tools across neuroscience, genomics, and computational biology. Yet hyperparameter selection remains largely unsystematic — researchers default to visual inspection or single-metric optimization, both of which risk silent geometric failure in one or more quality dimensions.

The **Hoegn Index** addresses this by implementing a maxmin robust optimization criterion: rather than maximizing any single quality metric, it identifies the parameter configuration where the *worst-performing* metric is at its best. This enforces a performance floor across all selected metrics simultaneously.

The repository also implements the **Hoegn Parametrization** — a logit-space interpolation method for sweeping structurally complex non-numeric parameters such as the DIABLO design matrix, which cannot be traversed by conventional grid search.

The full pipeline is demonstrated on a synthetic multimodal neurodevelopmental dataset integrating neuroimaging (EEG) and neuropsychological assessment data across ADHD, ASD, and normotypical groups.

---

## Repository Structure

```
hoegn-index/
├── README.md
├── LICENSE
├── requirements.txt                  ← Python dependencies
├── requirements_R.txt                ← R dependencies
│
├── R/
│   ├── 1 - deisgn matrices for DAIBLO code/ ← Empirical DIABLO design matrix computation
│   ├── 2 - Sweepers/                        ← Design matrix hyperparameter sweep
│   ├── 3 - Hoegn Index/                     ← Hoegn Index estimation
│   ├── 4 - final DIABLO extraction/         ← DIABLO embedding extraction
│   └── synthetic data/                      ← Synthetic data generation via synthpop
│
└── Python/
    ├── 1 - param_sweeps/             ← UMAP and PHATE hyperparameter sweeps
    ├── 2 - Hoegn Index/              ← Hoegn Index computation and diagnostics
    ├── 3 - analyses/                 ← Ground truth recovery validation
    ├── 4 - ground truth/             ← Downstream geometric and inferential validation

```

---

## Execution Order

> **Important:** The R pipeline must be completed before running any Python scripts. DIABLO outputs from R are the direct inputs to the UMAP and PHATE stages in Python.

### R — run in order

```
1. R/01_data_synthesis/
2. R/02_design_matrices/
3. R/03_diablo/
4. R/04_sweepers/
```

### Python — run in order

```
5. Python/01_sweepers/
6. Python/02_hoegn_index/
7. Python/03_ground_truth/
8. Python/04_validation/
9. Python/05_feature_interpretation/
10. Python/06_figures/
```

> **RDS file cleanup:** When running the DIABLO sweeper and Hoegn Index scripts, delete all intermediate `.rds` files at the end of each axis cycle before starting the next. Failure to do so will cause scripts to read stale metadata from a previous run. See inline comments in `R/04_sweepers/` for details.

---

## Installation

### Python

Requires Python 3.10. Install all dependencies via:

```bash
pip install -r requirements.txt
```

> Note: if you are using a conda environment (recommended), activate it before running the above.

### R

Requires R ≥ 4.2.3. Install all required packages by running the following in your R console:

```r
install.packages(c(
  "mixOmics",
  "synthpop",
  "ggplot2",
  "dplyr",
  "vegan",
  "reshape2",
  "here"
))
```

Full R session info including all dependency versions is available in `requirements_R.txt`.

---

## Data Availability

The synthetic dataset used in all analyses is publicly available on the Open Science Framework:

**OSF:** [https://doi.org/10.17605/OSF.IO/86YMB](https://doi.org/10.17605/OSF.IO/86YMB)

The synthetic dataset was generated from a proprietary clinical dataset collected at the University of Padova, Italy, using the `synthpop` package (v1.9.2) in R. The original clinical dataset is not publicly available and will be the subject of a forthcoming clinical publication.

---

## Citation

If you use the Hoegn Index or this pipeline in your work, please cite:

```
Hoegn, A. (2026). Validation of Low-Dimensional Embeddings for Supervised and
Unsupervised Dimensionality Reduction in Multimodal Datasets.
Manuscript in preparation.
```

---

## License

This repository is licensed under the **GNU General Public License v3.0**.
See [LICENSE](LICENSE) for full terms.

Any derivative work must be released under the same license.
