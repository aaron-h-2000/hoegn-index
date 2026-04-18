# phate_sweeper.py
# Sweeps the PHATE parameter space (knn × decay × gamma) over the neuroTests
# data and evaluates each embedding with four quality metrics. Results are
# collected in memory and written once at the end to avoid the race condition
# that arises from parallel appended CSV writes.

from pathlib import Path

import numpy as np
import pandas as pd
import phate
from joblib import Parallel, delayed
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


def compute_continuity(X, Y, n_neighbors = 5):
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
n_steps = 12

knn_vec   = np.unique(np.linspace(2, n_subjects - 2, n_steps).round().astype(int))
decay_vec = np.unique(np.linspace(5, 100,            n_steps).round().astype(int))
gamma_vec = np.linspace(-1, 1, n_steps)

# =============================================================================
# PER-COMBINATION WORKER FUNCTION
# =============================================================================

def compute_phate(knn, decay, gamma):
    """Fit PHATE and return a dict of quality metrics for one parameter set."""
    phate_model = phate.PHATE(
        knn          = knn,
        decay        = decay,
        gamma        = gamma,
        t            = "auto",
        n_components = 1,
        knn_dist     = "euclidean",
        random_state = random_state,
        n_jobs       = 1,          # must be 1 inside a Parallel call
        mds_solver   = "smacof",
        n_landmark   = None,
        mds_dist     = "euclidean",
        n_pca        = None,
        verbose      = 0,
    )
    embedding = phate_model.fit_transform(data)

    # Clamp k to a safe range for the neighbour-based metrics
    nn_metric = min(knn, n_subjects // 2)

    return {
        "knn":             knn,
        "decay":           decay,
        "gamma":           gamma,
        "trustworthiness": compute_trustworthiness(data, embedding, nn_metric),
        "continuity":      compute_continuity(data, embedding, nn_metric),
        "shepard":         compute_shepard_correlation(ref_embedding, embedding),
        "procrustes":      compute_procrustes_score(ref_embedding, embedding),
    }

# =============================================================================
# FULL PARAMETER GRID
# =============================================================================

param_combos = [
    (knn, decay, gamma)
    for knn   in knn_vec
    for decay in decay_vec
    for gamma in gamma_vec
]

results = Parallel(n_jobs = -1)(
    delayed(compute_phate)(knn, decay, gamma)
    for knn, decay, gamma in param_combos
)

output_csv = ROOT / "phate_sweep_results_neuroTests.csv"
pd.DataFrame(results).to_csv(output_csv, index = False)
print(f"Sweep complete. {len(results)} configurations saved to {output_csv}")