# permutation_bootstrapping_tests.py
# (filename corrected: "boostrapping" → "bootstrapping")
#
# Validates the group-separation structure of the joint manifold using:
#   - Observed MANOVA (Pillai's trace)
#   - Stratified bootstrap to estimate a 95 % CI around Pillai's trace
#   - Label-permutation test to obtain an empirical null distribution

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.multivariate.manova import MANOVA

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

# =============================================================================
# DATA
# =============================================================================

df           = pd.read_csv(ROOT / "results" / "final embedding" / "neurologicalData_PHATE_plus_neuroTests_UMAP_manifold.csv")
X_metric     = df[["Age", "neurologicalData", "neuroTests"]]
group_labels = df["group"].values

# =============================================================================
# PARAMETERS
# =============================================================================

n_bootstrap  = 1000
n_permutation = 1000
rng          = np.random.default_rng(seed = 13)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def stratified_resample(X_df, y_labels, rng):
    """Bootstrap resample within each group (stratified)."""
    rows, labels = [], []
    for g in np.unique(y_labels):
        idx     = np.where(y_labels == g)[0]
        sampled = rng.choice(idx, size = len(idx), replace = True)
        rows.append(X_df.iloc[sampled])
        labels.append(y_labels[sampled])
    return pd.concat(rows, ignore_index = True), np.concatenate(labels)


def compute_pillai(X_df, y_labels):
    """Return Pillai's trace from a one-way MANOVA of X_df ~ group."""
    data          = X_df.copy()
    data["group"] = y_labels
    formula       = " + ".join(X_df.columns) + " ~ group"
    res           = MANOVA.from_formula(formula, data = data).mv_test()
    return float(res.results["group"]["stat"].loc["Pillai's trace", "Value"])

# =============================================================================
# OBSERVED STATISTIC
# =============================================================================

observed_stat = compute_pillai(X_metric, group_labels)
print(f"Observed MANOVA Pillai's trace: {observed_stat:.4f}")

# =============================================================================
# BOOTSTRAP
# =============================================================================

bootstrap_stats = np.array([
    compute_pillai(*stratified_resample(X_metric, group_labels, rng))
    for _ in range(n_bootstrap)
])

ci_lower = float(np.percentile(bootstrap_stats, 2.5))
ci_upper = float(np.percentile(bootstrap_stats, 97.5))
print(f"Bootstrap 95 % CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# =============================================================================
# PERMUTATION TEST
# =============================================================================

permutation_stats = np.array([
    compute_pillai(X_metric, rng.permutation(group_labels))
    for _ in range(n_permutation)
])

p_value = float(np.mean(permutation_stats >= observed_stat))
print(f"Permutation p-value: {p_value:.4f}")

# =============================================================================
# PLOT — academic style
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Shared style parameters
HIST_KWARGS = dict(bins=30, edgecolor="white", linewidth=0.5, alpha=0.85)

# --- Left: Permutation null distribution ------------------------------------
ax = axes[0]
ax.hist(permutation_stats, color="#4393C3", **HIST_KWARGS)
ax.axvline(observed_stat, color="#B2182B", lw=1.8,
           label=f"Observed  ({observed_stat:.3f})")
ax.set_xlabel("Pillai's trace", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Permutation null distribution", fontsize=11, fontweight="bold", loc="left")
ax.annotate(f"p = {p_value:.3f}", xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.8))
ax.legend(frameon=True, framealpha=0.9, edgecolor="grey", fontsize=9)

# --- Right: Bootstrap sampling distribution ---------------------------------
ax = axes[1]
ax.hist(bootstrap_stats, color="#74C476", **HIST_KWARGS)
ax.axvline(ci_lower, color="#238B45", lw=1.4, ls="--",
           label=f"95 % CI  [{ci_lower:.3f}, {ci_upper:.3f}]")
ax.axvline(ci_upper, color="#238B45", lw=1.4, ls="--")
ax.axvline(observed_stat, color="#B2182B", lw=1.8,
           label=f"Observed  ({observed_stat:.3f})")
ax.set_xlabel("Pillai's trace", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Bootstrap sampling distribution", fontsize=11, fontweight="bold", loc="left")
ax.legend(frameon=True, framealpha=0.9, edgecolor="grey", fontsize=9)

# Shared theme
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("grey")
    ax.tick_params(labelsize=10, colors="dimgrey")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", color="gainsboro", lw=0.5, zorder=0)
    ax.grid(axis="x", visible=False)

fig.suptitle("MANOVA validation of group separation in the joint manifold",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
plt.savefig(ROOT / "permutation_bootstrap_manova.pdf", dpi=300, bbox_inches="tight")
plt.savefig(ROOT / "permutation_bootstrap_manova.png", dpi=300, bbox_inches="tight")
plt.show()