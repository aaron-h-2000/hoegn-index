# hoegn_index_phate.py
# Computes the Hoegn Index from the PHATE parameter sweep results and
# identifies the optimal (knn, decay, gamma) combination.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

# =============================================================================
# LOAD AND CLEAN SWEEP RESULTS
# =============================================================================

results = pd.read_csv(ROOT / "phate_sweep_results_neuroTests.csv")
results = results.dropna().reset_index(drop = True)

# Sort by sweep parameters so the x-axis follows a consistent order
results = results.sort_values(["knn", "decay", "gamma"]).reset_index(drop = True)

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

threshold  = 0.995 * A
idx_region = np.where(f_env >= threshold)[0]
x_region   = x_grid[idx_region]

uncertainty = float(x_region.max() - x_region.min())
confidence  = 1.0 - uncertainty / (x_grid.max() - x_grid.min())

# =============================================================================
# OUTPUT
# =============================================================================

hoegn_result = {
    "x_star":        x_star,
    "A":             A,
    "uncertainty":   uncertainty,
    "confidence":    confidence,
    "optimal_knn":   optimal_row["knn"],
    "optimal_decay": optimal_row["decay"],
    "optimal_gamma": optimal_row["gamma"],
}

print("\nHOEGN INDEX RESULT — PHATE")
for k, v in hoegn_result.items():
    print(f"  {k}: {v}")

pd.DataFrame([hoegn_result]).to_csv(ROOT / "phate_hoegn_index_result_neuroTests.csv", index = False)

# =============================================================================
# SECTION 7 — Save full sweep data to CSV
# =============================================================================

sweep_df = results.copy()
sweep_df["x_grid"]      = x_grid
sweep_df["hoegn_index"] = f_env

sweep_df.to_csv(ROOT / "phate_hoegn_sweep_metrics.csv", index=False)
print("Full sweep metrics saved to phate_hoegn_sweep_metrics.csv")

# =============================================================================
# SECTION 8 — Single-metric benchmark comparison
# For each metric, find what parameter combination it alone would select
# and record ALL other metric values at that point
# =============================================================================

param_cols = ["knn", "decay", "gamma"]

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
        "knn":               row["knn"],
        "decay":             row["decay"],
        "gamma":             row["gamma"],
        "trustworthiness":   f1[idx_best],
        "continuity":        f2[idx_best],
        "shepard":           f3[idx_best],
        "procrustes":        f4[idx_best],
        "hoegn_index":       f_env[idx_best],
    })

# Add Hoegn Index row
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
benchmark_df.to_csv(ROOT / "phate_hoegn_benchmark_comparison_neuroTests.csv", index=False)
print("Benchmark comparison saved to phate_hoegn_benchmark_comparison.csv")
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

fig, ax = plt.subplots(figsize = (9, 4.5))

# Confidence plateau shading
if len(x_region) > 1:
    ax.axvspan(x_region.min(), x_region.max(),
               color = PALETTE["Hoegn Index"], alpha = 0.08, zorder = 0,
               label = "99.5 % plateau")

# Metric lines
for label, values in [
    ("Trustworthiness", f1),
    ("Continuity",      f2),
    ("Shepard",         f3),
    ("Procrustes",      f4),
]:
    ax.plot(x_grid, values, color = PALETTE[label], lw = 1.2, label = label)

# Hoegn envelope
ax.plot(x_grid, f_env, color = PALETTE["Hoegn Index"],
        lw = 2.2, ls = "--", label = "Hoegn Index")

# Optimal point
ax.axvline(x_star, color = PALETTE["Hoegn Index"], lw = 1.0, ls = ":", zorder = 3)
ax.scatter([x_star], [A], color = PALETTE["Hoegn Index"],
           s = 60, zorder = 5, clip_on = False)
ax.annotate(r"$x^*$", xy = (x_star, A),
            xytext = (6, 6), textcoords = "offset points",
            fontsize = 9, color = PALETTE["Hoegn Index"], style = "italic")

# Axis labels and title
ax.set_xlabel(r"PHATE sweep trajectory  ($x$)", fontsize = 11)
ax.set_ylabel("Metric value", fontsize = 11)
ax.set_title("Hoegn Index — PHATE parameter sweep (neurologicalData)",
             fontsize = 12, fontweight = "bold", loc = "left")

# Legend
ax.legend(
    frameon = True, framealpha = 0.95, edgecolor = "grey",
    fontsize = 9, loc = "lower right",
    handlelength = 2.0
)

# Theme
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("grey")
ax.tick_params(axis = "both", labelsize = 10, colors = "dimgrey")
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis = "y", color = "gainsboro", lw = 0.5, zorder = 0)
ax.grid(axis = "x", visible = False)

fig.tight_layout()
plt.savefig(ROOT / "hoegn_index_phate_neurologicalData.pdf", dpi = 300, bbox_inches = "tight")
plt.savefig(ROOT / "hoegn_index_phate_neurologicalData.png", dpi = 300, bbox_inches = "tight")
plt.show()