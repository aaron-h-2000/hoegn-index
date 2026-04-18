# gt_hoegn_phate.py
# Hoegn Index computation, benchmark comparison, and ground truth evaluation
# for PHATE on manifold datasets.
#
# This is the complete Script 1 of the ground truth — it:
#   1. Computes the Hoegn Index and selects optimal parameters
#   2. Runs the benchmark comparison vs single-metric alternatives
#   3. Evaluates ground truth recovery via Spearman correlation with t
#
# Change DATASET_NAME to match the sweep results you want to evaluate:
#   "swiss_roll" | "s_curve" | "mammoth"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import phate
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

ROOT         = Path(r"C:\Users\aaron\Documents\Python\embedding paper")
DATASET_NAME = "mammoth"   # <-- change this for each run

# =============================================================================
# LOAD SWEEP RESULTS AND GROUND TRUTH DATA
# =============================================================================

results = pd.read_csv(ROOT / f"phate_sweep_results_{DATASET_NAME}.csv")
results = results.dropna().reset_index(drop=True)
results = results.sort_values(["knn", "decay", "gamma"]).reset_index(drop=True)

raw  = pd.read_csv(ROOT / "hoegn index empirical validation" / "ground truth" / f"ground_truth_{DATASET_NAME}.csv")
data = raw[["X1", "X2", "X3"]].to_numpy()
t    = raw["t"].to_numpy()

n = len(results)
x_grid = np.linspace(0, 10, n)

# =============================================================================
# HOEGN INDEX ENVELOPE
# =============================================================================

f1 = results["trustworthiness"].to_numpy(dtype=float)
f2 = results["continuity"].to_numpy(dtype=float)
f3 = results["shepard"].to_numpy(dtype=float)
f4 = results["procrustes"].to_numpy(dtype=float)

f_env    = np.minimum.reduce([f1, f2, f3, f4])
A        = float(np.max(f_env))
idx_star = int(np.argmax(f_env))
x_star   = x_grid[idx_star]

optimal_row = results.iloc[idx_star]

threshold   = 0.995 * A
idx_region  = np.where(f_env >= threshold)[0]
x_region    = x_grid[idx_region]
uncertainty = float(x_region.max() - x_region.min())
confidence  = 1.0 - uncertainty / (x_grid.max() - x_grid.min())

print(f"\nHOEGN INDEX — PHATE — {DATASET_NAME}")
print(f"  Optimal: knn={optimal_row['knn']}, "
      f"decay={optimal_row['decay']}, gamma={optimal_row['gamma']:.3f}")
print(f"  A={A:.4f}, confidence={confidence:.4f}")

# =============================================================================
# BENCHMARK COMPARISON vs SINGLE-METRIC ALTERNATIVES
# =============================================================================

single_metrics = {
    "argmax_trustworthiness": f1,
    "argmax_continuity":      f2,
    "argmax_shepard":         f3,
    "argmax_procrustes":      f4,
}

benchmark_rows = []
for method_name, metric_vec in single_metrics.items():
    idx_best = int(np.argmax(metric_vec))
    row      = results.iloc[idx_best]
    benchmark_rows.append({
        "selection_method":  method_name,
        "knn":               row["knn"],
        "decay":             row["decay"],
        "gamma":             row["gamma"],
        "trustworthiness":   f1[idx_best],
        "continuity":        f2[idx_best],
        "shepard":           f3[idx_best],
        "procrustes":        f4[idx_best],
        "hoegn_index":       f_env[idx_best],
    })

benchmark_rows.append({
    "selection_method":  "hoegn_index",
    "knn":               optimal_row["knn"],
    "decay":             optimal_row["decay"],
    "gamma":             optimal_row["gamma"],
    "trustworthiness":   f1[idx_star],
    "continuity":        f2[idx_star],
    "shepard":           f3[idx_star],
    "procrustes":        f4[idx_star],
    "hoegn_index":       A,
})

benchmark_df = pd.DataFrame(benchmark_rows)
benchmark_df.to_csv(ROOT / f"phate_gt_benchmark_{DATASET_NAME}.csv", index=False)
print("\nBenchmark comparison:")
print(benchmark_df[["selection_method", "trustworthiness", "continuity",
                     "shepard", "procrustes", "hoegn_index"]].to_string(index=False))

# =============================================================================
# SCRIPT 2 — GROUND TRUTH RECOVERY EVALUATION
# =============================================================================

def fit_phate_and_evaluate(knn, decay, gamma, label):
    try:
        phate_op  = phate.PHATE(
            n_components = 2,
            knn          = int(knn),
            decay        = int(decay),
            gamma        = float(gamma),
            verbose      = 0,
        )
        embedding     = phate_op.fit_transform(data)
        corr_dim1, _  = spearmanr(embedding[:, 0], t)
        corr_dim2, _  = spearmanr(embedding[:, 1], t)
        best_corr     = max(abs(corr_dim1), abs(corr_dim2))
        return {
            "selection_method":  label,
            "spearman_dim1":     round(corr_dim1, 4),
            "spearman_dim2":     round(corr_dim2, 4),
            "best_abs_spearman": round(best_corr, 4),
            "embedding":         embedding,
        }
    except Exception as e:
        print(f"  WARNING: {label} failed — {e}")
        return {
            "selection_method":  label,
            "spearman_dim1":     np.nan,
            "spearman_dim2":     np.nan,
            "best_abs_spearman": np.nan,
            "embedding":         None,
        }

print("\nEvaluating ground truth recovery for each selection method...")

gt_results = []
for _, brow in benchmark_df.iterrows():
    result = fit_phate_and_evaluate(
        brow["knn"], brow["decay"], brow["gamma"],
        brow["selection_method"],
    )
    gt_results.append(result)
    print(f"  {brow['selection_method']:<28} best |Spearman| = {result['best_abs_spearman']:.4f}")

gt_summary = pd.DataFrame([{k: v for k, v in r.items() if k != "embedding"}
                             for r in gt_results])
gt_summary.to_csv(ROOT / f"phate_gt_recovery_{DATASET_NAME}.csv", index=False)
print(f"\nGround truth recovery saved to phate_gt_recovery_{DATASET_NAME}.csv")

# =============================================================================
# PLOT — embedding coloured by t for each selection method
# =============================================================================

method_labels = {
    "argmax_trustworthiness": "Argmax Trustworthiness",
    "argmax_continuity":      "Argmax Continuity",
    "argmax_shepard":         "Argmax Shepard",
    "argmax_procrustes":      "Argmax Procrustes",
    "hoegn_index":            "Hoegn Index",
}

valid_results = [r for r in gt_results if r["embedding"] is not None]
fig, axes = plt.subplots(1, len(valid_results), figsize=(4 * len(valid_results), 4))
if len(valid_results) == 1:
    axes = [axes]

for ax, result in zip(axes, valid_results):
    emb = result["embedding"]
    sc  = ax.scatter(emb[:, 0], emb[:, 1], c=t, cmap="plasma",
                     s=8, alpha=0.8)
    label = method_labels.get(result["selection_method"], result["selection_method"])
    ax.set_title(f"{label}\n|ρ| = {result['best_abs_spearman']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("PHATE 1", fontsize=8)
    ax.set_ylabel("PHATE 2", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="t")

fig.suptitle(f"PHATE Ground Truth Recovery — {DATASET_NAME.replace('_', ' ').title()}",
             fontsize=12, fontweight="bold")
fig.tight_layout()
plt.savefig(ROOT / f"phate_gt_recovery_{DATASET_NAME}.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# HOEGN INDEX PLOT
# =============================================================================

PALETTE = {
    "Trustworthiness": "#2166AC",
    "Continuity":      "#4DAC26",
    "Shepard":         "#7B2D8B",
    "Procrustes":      "#E6851E",
    "Hoegn Index":     "#B2182B",
}

fig, ax = plt.subplots(figsize=(9, 4.5))

if len(x_region) > 1:
    ax.axvspan(x_region.min(), x_region.max(),
               color=PALETTE["Hoegn Index"], alpha=0.08, zorder=0)

for label, values in [("Trustworthiness", f1), ("Continuity", f2),
                       ("Shepard", f3), ("Procrustes", f4)]:
    ax.plot(x_grid, values, color=PALETTE[label], lw=1.2, label=label)

ax.plot(x_grid, f_env, color=PALETTE["Hoegn Index"],
        lw=2.2, ls="--", label="Hoegn Index")

ax.axvline(x_star, color=PALETTE["Hoegn Index"], lw=1.0, ls=":", zorder=3)
ax.scatter([x_star], [A], color=PALETTE["Hoegn Index"], s=60, zorder=5)
ax.annotate(r"$x^*$", xy=(x_star, A), xytext=(6, 6),
            textcoords="offset points", fontsize=9,
            color=PALETTE["Hoegn Index"], style="italic")

ax.set_xlabel("PHATE sweep trajectory", fontsize=11)
ax.set_ylabel("Metric value", fontsize=11)
ax.set_title(f"Hoegn Index — PHATE — {DATASET_NAME.replace('_', ' ').title()}",
             fontsize=12, fontweight="bold", loc="left")
ax.legend(frameon=True, framealpha=0.95, edgecolor="grey",
          fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="gainsboro", lw=0.5)
ax.grid(axis="x", visible=False)

fig.tight_layout()
plt.savefig(ROOT / f"phate_hoegn_index_{DATASET_NAME}.png", dpi=300, bbox_inches="tight")
plt.show()