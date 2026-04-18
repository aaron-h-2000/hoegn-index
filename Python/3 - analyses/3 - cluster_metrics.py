# cluster_metrics.py
# Validates the supervised manifold using clustering indices (silhouette,
# Davies-Bouldin, Calinski-Harabasz), PERMANOVA, and pairwise Hotelling T²
# tests across all group pairs.

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import f
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from skbio.stats.distance import DistanceMatrix, permanova

# NOTE: `Energy` from hyppo was imported previously but never used; removed.

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

df = pd.read_csv(ROOT / "results" / "final embedding" /"neurologicalData_PHATE_plus_neuroTests_UMAP_manifold.csv")

# =============================================================================
# HOTELLING T² HELPER
# =============================================================================

def hotelling_t2(X1, X2, regularization=1e-6):
    """
    Hotelling's T² statistic and corresponding F-test for two groups.

    Parameters
    ----------
    X1, X2        : (n1, p) and (n2, p) arrays for the two groups
    regularization: ridge term added to the pooled covariance to avoid
                    singularity with high-dimensional or collinear data

    Returns
    -------
    T2, F_stat, p_val
    """
    n1, p = X1.shape
    n2, _ = X2.shape

    mean_diff = X1.mean(axis=0) - X2.mean(axis=0)

    # Pooled covariance with regularisation
    S1 = np.cov(X1, rowvar=False)
    S2 = np.cov(X2, rowvar=False)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    Sp += np.eye(p) * regularization

    T2    = (n1 * n2) / (n1 + n2) * (mean_diff @ np.linalg.pinv(Sp) @ mean_diff)
    F_stat = ((n1 + n2 - p - 1) * T2) / ((n1 + n2 - 2) * p)
    p_val  = 1.0 - f.cdf(F_stat, p, n1 + n2 - p - 1)

    return T2, F_stat, p_val

# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_supervised_manifold(df, embedding_cols, label_col):
    """
    Compute clustering and group-separation metrics including pairwise
    Hotelling T².

    Returns
    -------
    summary_results : pd.Series  — global indices
    hotelling_df    : pd.DataFrame — pairwise Hotelling results
    """
    X            = df[embedding_cols].values
    labels       = df[label_col].values
    unique_groups = np.unique(labels)

    # --- Global clustering indices -------------------------------------------
    summary_results = pd.Series({
        "silhouette":         silhouette_score(X, labels),
        "davies_bouldin":     davies_bouldin_score(X, labels),
        "calinski_harabasz":  calinski_harabasz_score(X, labels),
    })

    # --- PERMANOVA -----------------------------------------------------------
    dist_matrix  = DistanceMatrix(cdist(X, X, metric="euclidean"))
    permanova_res = permanova(dist_matrix, labels, permutations=999)
    summary_results["permanova_F"] = permanova_res["test statistic"]
    summary_results["permanova_p"] = permanova_res["p-value"]

    # --- Pairwise Hotelling T² -----------------------------------------------
    pairwise_rows = []
    for g1, g2 in combinations(unique_groups, 2):
        X1, X2         = X[labels == g1], X[labels == g2]
        T2, F_stat, p_val = hotelling_t2(X1, X2)
        pairwise_rows.append({
            "group_1": g1,
            "group_2": g2,
            "T2":      T2,
            "F_stat":  F_stat,
            "p_value": p_val,
        })

    hotelling_df = pd.DataFrame(pairwise_rows)

    return summary_results, hotelling_df

# =============================================================================
# RUN
# =============================================================================

summary_results, hotelling_results = validate_supervised_manifold(
    df,
    embedding_cols=["Age", "neurologicalData", "neuroTests"],
    label_col="group",
)

print("Global clustering metrics:")
print(summary_results)
print("\nPairwise Hotelling T² tests:")
print(hotelling_results)