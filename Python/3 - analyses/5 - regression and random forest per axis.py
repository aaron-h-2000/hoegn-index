# regression_and_random_forest_per_axis.py
# Maps each UMAP embedding axis back to its source features via:
#   1. PCA dimensionality reduction of each feature block
#   2. Elastic Net regression (PCs → embedding axis) with back-projection
#      of coefficients onto original features
#   3. Random Forest regression (PCs → embedding axis) with feature importance
#      back-projected onto original features

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# =============================================================================
# PATHS
# =============================================================================

ROOT        = Path(r"C:\Users\aaron\Documents\Python\embedding paper")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok = True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_prefix(file_path, prefix):
    """Load a CSV and prepend a prefix to all column names."""
    return pd.read_csv(file_path).add_prefix(prefix)


def run_pca(df, prefix, n_components = 20):
    """Fit PCA and return (pca_object, scores_dataframe)."""
    pca    = PCA(n_components = n_components)
    scores = pd.DataFrame(
        pca.fit_transform(df),
        columns=[f"{prefix}_{i}" for i in range(n_components)],
        index = df.index,
    )
    return pca, scores


def run_elastic_net(X, y):
    """
    Fit an Elastic Net with cross-validated alpha / l1_ratio selection.

    Returns
    -------
    metrics      : DataFrame with R², alpha, and l1_ratio
    coefficients : Series of PC coefficients sorted by absolute value
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",   ElasticNetCV(
            l1_ratio   = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            cv         = 5,
            random_state = 42,
            max_iter   = 10_000,
        )),
    ])
    pipeline.fit(X, y)

    model = pipeline.named_steps["enet"]
    metrics = pd.DataFrame({
        "R2":      [pipeline.score(X, y)],
        "alpha":   [model.alpha_],
        "l1_ratio": [model.l1_ratio_],
    })
    coefficients = pd.Series(
        model.coef_, index = X.columns, name = "coefficient"
    ).sort_values(key = np.abs, ascending = False)

    return metrics, coefficients


def backproject_loadings(pca_obj, original_features, pc_values, name = "influence"):
    """
    Back-project PC-level coefficients/importances onto original features.

    Parameters
    ----------
    pca_obj          : fitted PCA object
    original_features: column names of the original feature matrix
    pc_values        : Series indexed by PC names, one value per component
    name             : name for the returned influence Series

    Returns
    -------
    loadings  : (n_features * n_components) DataFrame of PCA loadings
    projected : Series of per-feature influence scores
    """
    loadings = pd.DataFrame(
        pca_obj.components_.T,
        index   = original_features,
        columns = [f"PC{i + 1}" for i in range(pca_obj.n_components_)],
    )
    projected = pd.Series(
        loadings.values @ pc_values.values,
        index = original_features,
        name  = name,
    ).sort_values(ascending=False)

    return loadings, projected

# =============================================================================
# LOAD RAW FEATURE BLOCKS
# =============================================================================

# Neurological data
M1 = load_and_prefix(ROOT / "synthetic_data/synthetic neurologicalData/synthetic_Ap.csv",   "AP_")
M2 = load_and_prefix(ROOT / "synthetic_data/synthetic neurologicalData/synthetic_Coh.csv",  "COH_")
M3 = load_and_prefix(ROOT / "synthetic_data/synthetic neurologicalData/synthetic_Exc.csv",  "EXC_")
M4 = load_and_prefix(ROOT / "synthetic_data/synthetic neurologicalData/synthetic_Flex.csv", "FLEX_")
M5 = load_and_prefix(ROOT / "synthetic_data/synthetic neurologicalData/synthetic_Flu.csv",  "FLU_")

# Neuropsychological tests
M10 = load_and_prefix(ROOT / "synthetic_data/synthetic neuroTests/synthetic_CognitiveFlex.csv",    "COG_")
M11 = load_and_prefix(ROOT / "synthetic_data/synthetic neuroTests/synthetic_ConflictAccuracy.csv", "CA_")  # note capital C
M12 = load_and_prefix(ROOT / "synthetic_data/synthetic neuroTests/synthetic_ConflictSpe.csv",      "CS_")
M13 = load_and_prefix(ROOT / "synthetic_data/synthetic neuroTests/synthetic_ExecutiveFun.csv",     "EF_")

metadata     = pd.read_csv(ROOT / "synthetic_data/synthetic_behavioral.csv")
embeddings_df = pd.read_csv(ROOT / "results" / "final embedding" /"neurologicalData_PHATE_plus_neuroTests_UMAP_manifold.csv")

# =============================================================================
# CONCATENATE BLOCKS
# =============================================================================

neurologicalData = pd.concat([M1, M2, M3, M4, M5], axis = 1)
neuroTests       = pd.concat([M10, M11, M12, M13], axis = 1)

EXCLUDE_GROUPS = ["DSA", "epilessia"]
keep_mask = ~metadata["group"].isin(EXCLUDE_GROUPS)

neurologicalData = neurologicalData.loc[keep_mask].reset_index(drop=True)
neuroTests       = neuroTests.loc[keep_mask].reset_index(drop=True)
metadata         = metadata.loc[keep_mask].reset_index(drop=True)

# Embeddings are assumed to have been pre-filtered to matching rows
embeddings_df = embeddings_df.reset_index(drop=True)

# Fill any remaining missing values with column means
neurologicalData = neurologicalData.fillna(neurologicalData.mean())
neuroTests       = neuroTests.fillna(neuroTests.mean())

# =============================================================================
# PCA
# =============================================================================

neu_pca_obj,  neu_pca   = run_pca(neurologicalData, "neu")
quest_pca_obj, quest_pca = run_pca(neuroTests, "quest")

# =============================================================================
# REGRESSION MAP
# =============================================================================

regression_map = {
    "neurologicalData": {
        "X":                neu_pca,
        "y":                embeddings_df["neurologicalData"],
        "pca_obj":          neu_pca_obj,
        "original_features": neurologicalData.columns,
    },
    "neuroTests": {
        "X":                quest_pca,
        "y":                embeddings_df["neuroTests"],
        "pca_obj":          quest_pca_obj,
        "original_features": neuroTests.columns,
    },
}

# =============================================================================
# ELASTIC NET
# =============================================================================

print("\n" + "=" * 60)
print("ELASTIC NET RESULTS")
print("=" * 60)

for axis, cfg in regression_map.items():
    metrics, coefficients = run_elastic_net(cfg["X"], cfg["y"])
    loadings, influence   = backproject_loadings(
        cfg["pca_obj"], cfg["original_features"], coefficients
    )

    metrics.to_csv(RESULTS_DIR / f"{axis}_regression_metrics.csv", index=False)
    coefficients.to_csv(RESULTS_DIR / f"{axis}_pc_coefficients.csv")
    loadings.to_csv(RESULTS_DIR / f"{axis}_pca_loadings.csv")
    influence.to_csv(RESULTS_DIR / f"{axis}_feature_influence.csv")

    print(f"\n--- {axis} ---")
    print(metrics)
    print("\nTop 10 features:")
    print(influence.head(10))

# =============================================================================
# RANDOM FOREST
# =============================================================================

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

for axis, cfg in regression_map.items():
    X_rf = cfg["X"]
    y_rf = cfg["y"]

    rf = RandomForestRegressor(
        n_estimators     = 500,
        max_depth        = 15,
        min_samples_split = 3,
        min_samples_leaf  = 2,
        random_state     = 42,
        n_jobs           = -1,
    )

    # Cross-validate first, then refit on the full data to extract importances
    cv_scores = cross_val_score(rf, X_rf, y_rf, cv=5, scoring="r2")
    rf.fit(X_rf, y_rf)

    pc_importance = pd.Series(
        rf.feature_importances_, index=X_rf.columns, name="importance"
    ).sort_values(ascending=False)

    _, feature_importance = backproject_loadings(
        cfg["pca_obj"], cfg["original_features"],
        pc_importance, name="rf_importance"
    )

    rf_metrics = pd.DataFrame({
        "CV_R2_mean": [cv_scores.mean()],
        "CV_R2_std":  [cv_scores.std()],
    })

    rf_metrics.to_csv(RESULTS_DIR / f"{axis}_rf_metrics.csv", index=False)
    pc_importance.to_csv(RESULTS_DIR / f"{axis}_rf_pc_importance.csv")
    feature_importance.to_csv(RESULTS_DIR / f"{axis}_rf_feature_importance.csv")

    print(f"\n--- {axis} ---")
    print(rf_metrics)
    print("\nTop 10 PCs:")
    print(pc_importance.head(10))
    print("\nTop 10 original features:")
    print(feature_importance.head(10))

print("\nALL ANALYSES COMPLETE.")