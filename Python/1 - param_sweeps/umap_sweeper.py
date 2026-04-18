# umap_sweeper.py
# Sweeps the UMAP parameter space (distance metric × n_neighbors × min_dist)
# over the neurological data and evaluates each embedding with four quality
# metrics. Results are collected in memory and written once at the end to
# avoid the race condition that arises from parallel appended CSV writes.

from pathlib import Path

import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from umap import UMAP
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.spatial import procrustes

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

# =============================================================================
# QUALITY METRIC FUNCTIONS
# =============================================================================

def compute_trustworthiness(X, Y, n_neighbors = 5):
    return trustworthiness(X, Y, n_neighbors = n_neighbors)


def compute_continuity(X, Y, n_neighbors=5):
    """Fraction of true high-dim neighbours that are preserved in low-dim space."""
    n_samples = X.shape[0]

    neighbors_X = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)\
                                  .kneighbors(return_distance = False)[:, 1:]
    neighbors_Y = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Y)\
                                  .kneighbors(return_distance = False)[:, 1:]

    continuity_sum = sum(
        len(set(neighbors_X[i]) & set(neighbors_Y[i])) / n_neighbors
        for i in range(n_samples)
    )
    return continuity_sum / n_samples


def compute_shepard_correlation(X_ref, Y):
    """Spearman correlation between pairwise distances in high- and low-dim space."""
    d_high = squareform(pdist(X_ref, metric = "euclidean"))
    d_low  = squareform(pdist(Y,     metric = "euclidean"))
    corr, _ = spearmanr(d_high.flatten(), d_low.flatten())
    return corr


def compute_procrustes_score(X_ref, Y):
    """1 Procrustes disparity (higher = better alignment)."""
    _, _, disparity = procrustes(X_ref, Y)
    return 1.0 - disparity

# =============================================================================
# DATA AND PARAMETERS
# =============================================================================

data = pd.read_csv(ROOT / "results" / "axes" / "neuroTests_1axis_forPy.csv")
n_subjects = data.shape[0]

# Reference embedding for Shepard / Procrustes comparisons
ref_embedding = pd.read_csv(ROOT / "results" / "axes" / "neuroTests_1axis_noPy.csv").to_numpy()

random_state = 13

# Fixed sweep resolution
n_steps = 24

n_neighbors_vec = np.unique(
    np.linspace(2, n_subjects - 1, n_steps).round().astype(int)
)
min_dist_vec = np.linspace(0.0, 1.0, n_steps)
metrics_vec  = ["euclidean", "manhattan", "chebyshev"]

# =============================================================================
# PER-COMBINATION WORKER FUNCTION
# =============================================================================

def compute_umap(nn, md, metric):
    """Fit UMAP and return a dict of quality metrics for one parameter set."""
    umap_model = UMAP(
        n_neighbors  = nn,
        min_dist     = md,
        n_components = 1,
        metric       = metric,
        random_state = random_state,
        init         = "random",
        n_jobs       = 1,          # must be 1 inside a Parallel call
    )
    embedding = umap_model.fit_transform(data)

    # Clamp k to a safe range for the neighbour-based metrics
    nn_metric = min(nn, n_subjects // 2)

    return {
        "metric":          metric,
        "n_neighbors":     nn,
        "min_dist":        md,
        "trustworthiness": compute_trustworthiness(data, embedding, nn_metric),
        "continuity":      compute_continuity(data, embedding, nn_metric),
        "shepard":         compute_shepard_correlation(ref_embedding, embedding),
        "procrustes":      compute_procrustes_score(ref_embedding, embedding),
    }

# =============================================================================
# FULL PARAMETER GRID
# =============================================================================

param_combos = [
    (metric, nn, md)
    for metric in metrics_vec
    for nn in n_neighbors_vec
    for md in min_dist_vec
]

results = Parallel(n_jobs = -1)(
    delayed(compute_umap)(nn, md, metric)
    for metric, nn, md in param_combos
)

output_csv = ROOT / "umap_sweep_results_neuroTests.csv"
pd.DataFrame(results).to_csv(output_csv, index = False)
print(f"Sweep complete. {len(results)} configurations saved to {output_csv}")