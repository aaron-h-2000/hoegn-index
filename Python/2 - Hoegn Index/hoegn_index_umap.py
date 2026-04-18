# hoegn_index_umap.py
# Computes the Hoegn Index from the UMAP parameter sweep results and
# identifies the optimal (metric, n_neighbors, min_dist) combination.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

# =============================================================================
# LOAD AND CLEAN SWEEP RESULTS
# =============================================================================

results = pd.read_csv(ROOT / "umap_sweep_results_neurologicalData.csv")

metric_cols = ["trustworthiness", "continuity", "shepard", "procrustes"]
for col in metric_cols:
    results[col] = pd.to_numeric(results[col], errors = "coerce")

results = results.dropna(subset = metric_cols).reset_index(drop = True)

# Sort by distance metric → n_neighbors → min_dist so the x-axis reflects
# the structured sweep order (Euclidean / Manhattan / Chebyshev sections)
metric_order = ["euclidean", "manhattan", "chebyshev"]
results["metric"] = pd.Categorical(
    results["metric"], categories = metric_order, ordered = True
)
results = results.sort_values(["metric", "n_neighbors", "min_dist"]).reset_index(drop = True)

# Sequential x-axis [0, 10] for display only
n      = len(results)
x_grid = np.linspace(0, 10, n)

# =============================================================================
# METRIC ARRAYS AND HOEGN ENVELOPE
# =============================================================================

f1 = results["trustworthiness"].to_numpy(dtype = float)
f2 = results["continuity"].to_numpy(dtype = float)
f3 = results["shepard"].to_numpy(dtype = float)
f4 = results["procrustes"].to_numpy(dtype = float)

# Hoegn envelope: pointwise minimum across all four metrics
f_env = np.minimum.reduce([f1, f2, f3, f4])

A        = float(np.max(f_env))
idx_star = int(np.argmax(f_env))
x_star   = x_grid[idx_star]

optimal_row = results.iloc[idx_star]

# =============================================================================
# CONFIDENCE / PLATEAU WIDTH
# =============================================================================

threshold  = 0.9995 * A
idx_region = np.where(f_env >= threshold)[0]
x_region   = x_grid[idx_region]

uncertainty = float(x_region.max() - x_region.min())
confidence  = 1.0 - uncertainty / (x_grid.max() - x_grid.min())

# =============================================================================
# OUTPUT
# =============================================================================

hoegn_result = {
    "x_star":              x_star,
    "A":                   A,
    "uncertainty":         uncertainty,
    "confidence":          confidence,
    "optimal_metric":      optimal_row["metric"],
    "optimal_n_neighbors": optimal_row["n_neighbors"],
    "optimal_min_dist":    optimal_row["min_dist"],
}

print("\nHOEGN INDEX RESULT — UMAP")
for k, v in hoegn_result.items():
    print(f"  {k}: {v}")

pd.DataFrame([hoegn_result]).to_csv(ROOT / "hoegn_index_result_umap_neurologicalData.csv", index=False)

# =============================================================================
# SECTION 7 — Save full sweep data to CSV
# =============================================================================

sweep_df = results.copy()
sweep_df["x_grid"]      = x_grid
sweep_df["hoegn_index"] = f_env

sweep_df.to_csv(ROOT / "umap_hoegn_sweep_metrics.csv", index=False)
print("Full sweep metrics saved to umap_hoegn_sweep_metrics.csv")

# =============================================================================
# SECTION 8 — Single-metric benchmark comparison
# For each metric, find what parameter combination it alone would select
# and record ALL other metric values at that point
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
    row = results.iloc[idx_best]
    benchmark_rows.append({
        "selection_method":  method_name,
        "distance_metric":   row["metric"],
        "n_neighbors":       row["n_neighbors"],
        "min_dist":          row["min_dist"],
        "trustworthiness":   f1[idx_best],
        "continuity":        f2[idx_best],
        "shepard":           f3[idx_best],
        "procrustes":        f4[idx_best],
        "hoegn_index":       f_env[idx_best],
    })

# Add Hoegn Index row
benchmark_rows.append({
    "selection_method":  "hoegn_index",
    "distance_metric":   optimal_row["metric"],
    "n_neighbors":       optimal_row["n_neighbors"],
    "min_dist":          optimal_row["min_dist"],
    "trustworthiness":   f1[idx_star],
    "continuity":        f2[idx_star],
    "shepard":           f3[idx_star],
    "procrustes":        f4[idx_star],
    "hoegn_index":       A,
})

benchmark_df = pd.DataFrame(benchmark_rows)
benchmark_df.to_csv(ROOT / "umap_hoegn_benchmark_comparison_neurologicalData.csv", index=False)
print("Benchmark comparison saved to umap_hoegn_benchmark_comparison.csv")
print(benchmark_df.to_string(index=False))

# =============================================================================
# PLOT — academic style
# =============================================================================

PALETTE = {
    "Trustworthiness": "#2166AC",
    "Continuity":      "#4DAC26",
    "Shepard":         "#7B2D8B",
    "Procrustes":      "#E6851E",
    "Hoegn Index":     "#B2182B",
}

# Section boundaries for the three distance metrics
n_per_section = n // 3
section_cuts  = [x_grid[0],
                 x_grid[n_per_section],
                 x_grid[2 * n_per_section],
                 x_grid[-1]]
section_mids  = [(section_cuts[i] + section_cuts[i + 1]) / 2 for i in range(3)]

fig, ax = plt.subplots(figsize=(9, 4.5))

# Confidence plateau shading
if len(x_region) > 1:
    ax.axvspan(x_region.min(), x_region.max(),
               color=PALETTE["Hoegn Index"], alpha=0.08, zorder=0,
               label="99.95 % plateau")

# Metric lines
for label, values in [
    ("Trustworthiness", f1),
    ("Continuity",      f2),
    ("Shepard",         f3),
    ("Procrustes",      f4),
]:
    ax.plot(x_grid, values, color=PALETTE[label], lw=1.2, label=label)

# Hoegn envelope
ax.plot(x_grid, f_env, color=PALETTE["Hoegn Index"],
        lw=2.2, ls="--", label="Hoegn Index")

# Distance-metric section separators
for cut in section_cuts[1:-1]:
    ax.axvline(cut, color="grey", lw=0.8, ls="--", alpha=0.5)

# Optimal point
ax.axvline(x_star, color=PALETTE["Hoegn Index"], lw=1.0, ls=":", zorder=3)
ax.scatter([x_star], [A], color=PALETTE["Hoegn Index"],
           s=60, zorder=5, clip_on=False)
ax.annotate(r"$x^*$", xy=(x_star, A),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9, color=PALETTE["Hoegn Index"], style="italic")

# Custom x-axis ticks at section midpoints
ax.set_xticks(section_mids)
ax.set_xticklabels(["Euclidean", "Manhattan", "Chebyshev"], fontsize=10)
ax.xaxis.set_minor_locator(ticker.NullLocator())

# Axis labels and title
ax.set_xlabel("Distance metric  (sweep trajectory)", fontsize=11)
ax.set_ylabel("Metric value", fontsize=11)
ax.set_title("Hoegn Index — UMAP parameter sweep (neuroTests data)",
             fontsize=12, fontweight="bold", loc="left")

# Legend
ax.legend(
    frameon=True, framealpha=0.95, edgecolor="grey",
    fontsize=9, loc="lower right",
    handlelength=2.0
)

# Theme
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("grey")
ax.tick_params(axis="both", labelsize=10, colors="dimgrey")
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis="y", color="gainsboro", lw=0.5, zorder=0)
ax.grid(axis="x", visible=False)

fig.tight_layout()
plt.savefig(ROOT / "hoegn_index_umap_neuroTests.pdf", dpi=300, bbox_inches="tight")
plt.savefig(ROOT / "hoegn_index_umap_neuroTests.png", dpi=300, bbox_inches="tight")
plt.show()