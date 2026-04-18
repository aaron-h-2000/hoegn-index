# umap_and_phate_for_DIABLO_exports.py
# Applies the optimal UMAP and PHATE hyperparameters (identified by the Hoegn
# Index sweep) to produce the final single-axis embeddings for downstream use.

from pathlib import Path

import numpy as np
import pandas as pd
import phate
import umap

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(ROOT / "results" / "axes" / "neurologicalData_1axis_forPy.csv")

# =============================================================================
# UMAP — optimal hyperparameters from Hoegn Index sweep
# =============================================================================

umap_model = umap.UMAP(
    n_neighbors  = 130,
    min_dist     = 0.174,
    n_components = 1,
    metric       = "chebyshev",
    random_state = 13,
)

umap_results = umap_model.fit_transform(df)
print("UMAP output shape:", umap_results.shape)

umap_df = pd.DataFrame(umap_results, columns = ["UMAP_comp1"])

umap_df.to_csv(ROOT / "results" / "final axes" / "neurologicalData_1axis_UMAP.csv", index = False)

# =============================================================================
# PHATE — optimal hyperparameters from Hoegn Index sweep
# =============================================================================

phate_model = phate.PHATE(
    knn          = 103,
    decay        = 14,
    gamma        = -0.091,
    t            = "auto",
    n_components = 1,
    knn_dist     = "euclidean",
    random_state = 13,
)

phate_results = phate_model.fit_transform(df)
print("PHATE output shape:", phate_results.shape)

phate_df = pd.DataFrame(phate_results, columns = ["PHATE_comp1"])

phate_df.to_csv(ROOT / "results" / "final axes" / "neurologicalData_1axis_PHATE.csv", index = False)