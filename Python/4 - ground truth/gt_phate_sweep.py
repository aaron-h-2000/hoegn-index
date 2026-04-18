# gt_phate_sweep.py
# PHATE parameter sweep for ground truth manifold datasets.
# Replaces the clinical sweep script — no DIABLO, no block structure.
# Input:  a ground truth CSV with columns X1, X2, X3, t
# Output: phate_sweep_results_{dataset_name}.csv
#
# Change DATASET_NAME to match the file you want to sweep:
#   "swiss_roll" | "s_curve" | "mammoth"

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import phate
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
t    = raw["t"].to_numpy()

n_subjects = data.shape[0]

# Reference for Shepard and Procrustes is the original 3D space
ref_3d = data.copy()
ref_2d = PCA(n_components=2).fit_transform(ref_3d)

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
    _, _, disparity = procrustes(X_ref_2d, Y)
    return 1.0 - disparity

# =============================================================================
# SWEEP PARAMETERS
# =============================================================================

knn_vec = np.unique(np.linspace(2, min(n_subjects // 2 - 1, 150), 10).round().astype(int))
decay_vec = np.linspace(1, 100, 10).round().astype(int)
gamma_vec = np.linspace(-1, 1, 5).round(2)

# =============================================================================
# WORKER FUNCTION
# =============================================================================

def compute_phate(knn, decay, gamma):
    try:
        phate_op  = phate.PHATE(
            n_components = 2,
            knn          = int(knn),
            decay        = int(decay),
            gamma        = float(gamma),
            n_jobs       = 1,
            verbose      = 0,
        )
        embedding = phate_op.fit_transform(data)
        nn_metric = min(int(knn), n_subjects // 2)

        return {
            "knn":             knn,
            "decay":           decay,
            "gamma":           gamma,
            "trustworthiness": compute_trustworthiness(data, embedding, nn_metric),
            "continuity":      compute_continuity(data, embedding, nn_metric),
            "shepard":         compute_shepard_correlation(ref_3d, embedding),
            "procrustes":      compute_procrustes_score(ref_2d, embedding),
        }
    except Exception as e:
        return {
            "knn": knn, "decay": decay, "gamma": gamma,
            "trustworthiness": np.nan, "continuity": np.nan,
            "shepard": np.nan, "procrustes": np.nan,
        }

# =============================================================================
# RUN SWEEP
# =============================================================================

param_combos = [
    (knn, decay, gamma)
    for knn   in knn_vec
    for decay  in decay_vec
    for gamma  in gamma_vec
]

print(f"Running PHATE sweep for {DATASET_NAME} — {len(param_combos)} configurations...")

results = Parallel(n_jobs=-1)(
    delayed(compute_phate)(knn, decay, gamma)
    for knn, decay, gamma in param_combos
)

output_csv = ROOT / f"phate_sweep_results_{DATASET_NAME}.csv"
pd.DataFrame(results).dropna().to_csv(output_csv, index=False)
print(f"Sweep complete. Saved to {output_csv}")
