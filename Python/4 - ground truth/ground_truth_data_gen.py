# ground_truth_data_generation.py
# Generates three ground truth manifold datasets for validating the Hoegn Index
# parameter selection against single-metric alternatives.
#
# Outputs one CSV per dataset, formatted to match the existing sweep scripts:
#   - ground_truth_swiss_roll.csv
#   - ground_truth_s_curve.csv
#   - ground_truth_mammoth.csv
#
# Each CSV contains:
#   - X1, X2, X3: the high-dimensional coordinates fed into UMAP/PHATE
#   - t:          the ground truth manifold position (used in Script 2 evaluation)

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll, make_s_curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

N_SAMPLES = 300  # consistent sample size across all datasets
NOISE     = 0.1   
SEED      = 13    

# =============================================================================
# 1. SWISS ROLL
# =============================================================================

X_swiss, t_swiss = make_swiss_roll(n_samples=N_SAMPLES, noise=NOISE,
                                    random_state=SEED)

df_swiss = pd.DataFrame(X_swiss, columns=["X1", "X2", "X3"])
df_swiss["t"] = t_swiss

df_swiss.to_csv(ROOT / "ground_truth_swiss_roll.csv", index=False)
print(f"Swiss roll saved: {df_swiss.shape} — t range [{t_swiss.min():.2f}, {t_swiss.max():.2f}]")

# =============================================================================
# 2. S-CURVE
# =============================================================================

X_scurve, t_scurve = make_s_curve(n_samples=N_SAMPLES, noise=NOISE,
                                    random_state=SEED)

df_scurve = pd.DataFrame(X_scurve, columns=["X1", "X2", "X3"])
df_scurve["t"] = t_scurve

df_scurve.to_csv(ROOT / "ground_truth_s_curve.csv", index=False)
print(f"S-curve saved:    {df_scurve.shape} — t range [{t_scurve.min():.2f}, {t_scurve.max():.2f}]")

# =============================================================================
# 3. MAMMOTH
# Fetched from the PAIR-code/understanding-umap repository.
# Falls back to a locally generated torus knot if the download fails.
# =============================================================================

try:
    import urllib.request, json

    url = "https://raw.githubusercontent.com/PAIR-code/understanding-umap/master/raw_data/mammoth_3d.json"
    with urllib.request.urlopen(url, timeout=15) as response:
        mammoth_raw = json.loads(response.read().decode())

    # Raw data is a list of [x, y, z] triplets — subsample to N_SAMPLES
    mammoth_array = np.array(mammoth_raw)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(mammoth_array), size=N_SAMPLES, replace=False)
    mammoth_array = mammoth_array[idx]

    # Mammoth has no scalar t — we use distance along the first principal
    # component as a proxy ground truth (standard approach for point clouds)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    t_mammoth = pca.fit_transform(mammoth_array).ravel()

    df_mammoth = pd.DataFrame(mammoth_array, columns=["X1", "X2", "X3"])
    df_mammoth["t"] = t_mammoth

    df_mammoth.to_csv(ROOT / "ground_truth_mammoth.csv", index=False)
    print(f"Mammoth saved:    {df_mammoth.shape} — t range [{t_mammoth.min():.2f}, {t_mammoth.max():.2f}]")

except Exception as e:
    print(f"Mammoth download failed ({e}) — generating torus knot fallback.")

    # Torus knot: a well-known 3D curve with clear 1D ground truth
    t_knot  = np.linspace(0, 2 * np.pi, N_SAMPLES)
    p, q    = 2, 3          # standard (2,3) torus knot
    r       = 2
    X_knot  = np.column_stack([
        (r + np.cos(q * t_knot)) * np.cos(p * t_knot),
        (r + np.cos(q * t_knot)) * np.sin(p * t_knot),
        np.sin(q * t_knot)
    ])
    X_knot += np.random.default_rng(SEED).normal(0, NOISE, X_knot.shape)

    df_mammoth = pd.DataFrame(X_knot, columns=["X1", "X2", "X3"])
    df_mammoth["t"] = t_knot

    df_mammoth.to_csv(ROOT / "ground_truth_mammoth.csv", index=False)
    print(f"Torus knot saved: {df_mammoth.shape} — t range [{t_knot.min():.2f}, {t_knot.max():.2f}]")

# =============================================================================
# QUICK VISUALISATION — confirm the datasets look correct before sweeping
# =============================================================================

fig = plt.figure(figsize=(15, 4))

datasets = [
    ("Swiss Roll",  X_swiss,          t_swiss),
    ("S-Curve",     X_scurve,         t_scurve),
    ("Mammoth/Knot",df_mammoth[["X1","X2","X3"]].values, df_mammoth["t"].values),
]

for i, (title, X, t) in enumerate(datasets, 1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                    c=t, cmap="plasma", s=8, alpha=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X1", fontsize=8)
    ax.set_ylabel("X2", fontsize=8)
    ax.set_zlabel("X3", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="t (ground truth)")

fig.suptitle("Ground Truth Manifolds — coloured by t", fontsize=13,
             fontweight="bold", y=1.02)
fig.tight_layout()
plt.savefig(ROOT / "ground_truth_datasets_overview.png", dpi=300,
            bbox_inches="tight")
plt.show()

print("\nAll datasets generated successfully.")
print("Next step: feed each CSV into your UMAP and PHATE sweep scripts.")
print("Remember to use only X1, X2, X3 as input features — exclude the t column.")