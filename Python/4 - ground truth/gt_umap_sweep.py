# gt_umap_sweep.py
# UMAP parameter sweep for ground truth manifold datasets.
# Replaces the clinical sweep script — no DIABLO, no block structure.
# Input:  a ground truth CSV with columns X1, X2, X3, t
# Output: umap_sweep_results_{dataset_name}.csv
#
# Change DATASET_NAME to match the file you want to sweep:
#   "swiss_roll" | "s_curve" | "mammoth"

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from umap import UMAP
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.spatial import procrustes

ROOT         = Path(r"C:\Users\aaron\Documents\Python\embedding paper")
DATASET_NAME = "mammoth"   # <-- change this for each run

# =============================================================================
# LOAD DATA — exclude t, keep only X1 X2 X3
# =============================================================================

raw  = pd.read_csv(ROOT / "hoegn index empirical validation" / "ground truth" / f"ground_truth_{DATASET_NAME}.csv")
data = raw[["X1", "X2", "X3"]].to_numpy()
t    = raw["t"].to_numpy()          # saved but not used during sweep

n_subjects = data.shape[0]

# Reference for Shepard and Procrustes is the original 3D space
# For Procrustes we also need a 2D version since UMAP outputs 2D
ref_3d = data.copy()
ref_2d = PCA(n_components=2).fit_transform(ref_3d)

random_state = 13

# =============================================================================
# QUALITY METRIC FUNCTIONS
# =============================================================================

def compute_trustworthiness(X, Y, n_neighbors=5):
    return trustworthiness(X, Y, n_neighbors=n_neighbors)

def compute_continuity(X, Y, n_neighbors=5):
    n_samples   = X.shape[0]
    neighbors_X = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)\
                                   .kneighbors(return_distance=False)[:, 1:]
    neighbors_Y = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Y)\
                                   .kneighbors(return_distance=False)[:, 1:]
    continuity_sum = sum(
        len(set(neighbors_X[i]) & set(neighbors_Y[i])) / n_neighbors
        for i in range(n_samples)
    )
    return continuity_sum / n_samples

def compute_shepard_correlation(X_ref, Y):
    d_high = squareform(pdist(X_ref, metric="euclidean"))
    d_low  = squareform(pdist(Y,     metric="euclidean"))
    corr, _ = spearmanr(d_high.flatten(), d_low.flatten())
    return corr

def compute_procrustes_score(X_ref_2d, Y):
    # Both must be 2D — X_ref_2d is already PCA-reduced above
    _, _, disparity = procrustes(X_ref_2d, Y)
    return 1.0 - disparity

# =============================================================================
# SWEEP PARAMETERS
# =============================================================================

n_steps          = 15
n_neighbors_vec = np.unique(
    np.linspace(2, n_subjects // 2 - 1, n_steps).round().astype(int)
)
min_dist_vec     = np.linspace(0.0, 1.0, n_steps)
metrics_vec      = ["euclidean", "manhattan", "chebyshev"]

# =============================================================================
# WORKER FUNCTION
# =============================================================================

def compute_umap(nn, md, metric):
    try:
        umap_model = UMAP(
            n_neighbors  = nn,
            min_dist     = md,
            n_components = 2,
            metric       = metric,
            random_state = random_state,
            init         = "random",
            n_jobs       = 1,
        )
        embedding  = umap_model.fit_transform(data)
        nn_metric  = min(nn, n_subjects // 2 - 1)
        return {
            "metric":          metric,
            "n_neighbors":     nn,
            "min_dist":        md,
            "trustworthiness": compute_trustworthiness(data, embedding, nn_metric),
            "continuity":      compute_continuity(data, embedding, nn_metric),
            "shepard":         compute_shepard_correlation(ref_3d, embedding),
            "procrustes":      compute_procrustes_score(ref_2d, embedding),
        }
    except Exception as e:
        return {
            "metric": metric, "n_neighbors": nn, "min_dist": md,
            "trustworthiness": np.nan, "continuity": np.nan,
            "shepard": np.nan, "procrustes": np.nan,
        }

# =============================================================================
# RUN SWEEP
# =============================================================================

param_combos = [
    (metric, nn, md)
    for metric in metrics_vec
    for nn     in n_neighbors_vec
    for md     in min_dist_vec
]

print(f"Running UMAP sweep for {DATASET_NAME} — {len(param_combos)} configurations...")

results = Parallel(n_jobs=-1)(
    delayed(compute_umap)(nn, md, metric)
    for metric, nn, md in param_combos
)

output_csv = ROOT / f"umap_sweep_results_{DATASET_NAME}.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Sweep complete. {len(results)} configurations saved to {output_csv}")
