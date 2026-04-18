# plotting_of_features.py
# Loads Elastic Net and Random Forest feature-importance results for each
# embedding axis and produces publication-quality summary plots.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# =============================================================================
# SETTINGS
# =============================================================================

ROOT    = Path(r"C:\Users\aaron\Documents\Python\embedding paper")
RESULTS = ROOT / "results" / "ENRF"
FIGURES = ROOT / "results" / "ENRF" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
AXES    = ["neurologicalData", "neuroTests"]
TOP_N   = 20

# Consistent palette across all figures
BAR_COLORS = {
    "EN_influence":  "#2166AC",
    "RF_importance": "#E6851E",
    "Consensus":     "#4DAC26",
}

POSITIVE_COLOR = "#2166AC"   # blue  — positive influence
NEGATIVE_COLOR = "#B2182B"   # red   — negative influence

# =============================================================================
# SAVE HELPER
# =============================================================================

def save_fig(fig, name):
    """Save figure to FIGURES directory and print confirmation."""
    out_path = FIGURES / f"{name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out_path}")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_axis_results(axis):
    """Load all result CSVs for one embedding axis; missing files return None."""
    files = {
        "feature_influence":     RESULTS / f"{axis}_feature_influence.csv",
        "pc_coefficients":       RESULTS / f"{axis}_pc_coefficients.csv",
        "rf_feature_importance": RESULTS / f"{axis}_rf_feature_importance.csv",
        "pca_loadings":          RESULTS / f"{axis}_pca_loadings.csv",
        "rf_pc_importance":      RESULTS / f"{axis}_rf_pc_importance.csv",
        "metrics":               RESULTS / f"{axis}_regression_metrics.csv",
        "rf_metrics":            RESULTS / f"{axis}_rf_metrics.csv",
    }
    loaded = {}
    for key, path in files.items():
        try:
            loaded[key] = pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            print(f"  Warning: {key} not found for axis '{axis}'")
            loaded[key] = None
    return loaded

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary_stats(data):
    """Merge EN and RF importances into a consensus score."""
    summary = pd.DataFrame()

    if data["feature_influence"] is not None:
        summary["EN_influence"] = data["feature_influence"].squeeze()

    if data["rf_feature_importance"] is not None:
        summary["RF_importance"] = data["rf_feature_importance"].squeeze()

    if {"EN_influence", "RF_importance"}.issubset(summary.columns):
        norm_en = summary["EN_influence"].abs() / summary["EN_influence"].abs().sum()
        norm_rf = summary["RF_importance"].abs() / summary["RF_importance"].abs().sum()
        summary["Consensus"] = (norm_en + norm_rf) / 2

        corr = summary["EN_influence"].corr(summary["RF_importance"])
        print(f"  EN vs RF correlation: {corr:.3f}")

    if (data["metrics"] is not None) and ("R2" in data["metrics"].columns) \
            and ("Consensus" in summary.columns):
        r2 = data["metrics"]["R2"].mean()
        summary["Relative_Impact"] = summary["Consensus"] * r2

    if "Consensus" in summary.columns:
        summary = summary.sort_values("Consensus", ascending=False)

    return summary


def compute_consensus(feature_influence, rf_importance):
    norm_en = feature_influence.abs() / feature_influence.abs().sum()
    norm_rf = rf_importance.abs() / rf_importance.abs().sum()
    return ((norm_en + norm_rf) / 2).sort_values(ascending=False)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _apply_bar_theme(ax, xlabel=""):
    """Apply consistent academic styling to a horizontal bar chart."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("grey")
    ax.tick_params(labelsize=9, colors="dimgrey")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="x", color="gainsboro", lw=0.5, zorder=0)
    ax.grid(axis="y", visible=False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)


def plot_diverging_influence(series, title, top_n=TOP_N, save_name=None):
    """
    Diverging horizontal bar chart showing positive (blue) and negative (red)
    feature influence — replicates the style of original Figs 19 and 20.
    Sorted by absolute value; direction preserved.
    """
    plot_data = series.abs().sort_values(ascending=False).head(top_n)
    plot_data = series.loc[plot_data.index].sort_values()   # keep sign, sort asc

    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in plot_data.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_data.index, plot_data.values, color=colors, edgecolor="white")
    ax.axvline(0, color="grey", lw=0.8, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    _apply_bar_theme(ax, xlabel="Elastic Net influence")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, label="Positive influence"),
        Patch(facecolor=NEGATIVE_COLOR, label="Negative influence"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=True,
              framealpha=0.9, edgecolor="grey", loc="lower right")

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    plt.show()
    plt.close(fig)


def plot_top_series(series, title, top_n=TOP_N, save_name=None):
    """Horizontal bar chart of the top-N absolute values in a Series."""
    plot_data = series.abs().sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_data.index, plot_data.values, color="#4393C3", edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    _apply_bar_theme(ax, xlabel="Absolute value / importance")
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    plt.show()
    plt.close(fig)


def plot_top_features_pub(summary_stats, axis_name, top_n=TOP_N, save_name=None):
    """
    Side-by-side normalised bars for EN influence, RF importance, and Consensus.
    """
    cols_present = [c for c in ["EN_influence", "RF_importance", "Consensus"]
                    if c in summary_stats.columns]
    if not cols_present:
        return

    top_features = (
        summary_stats["Consensus"].abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        if "Consensus" in summary_stats.columns
        else summary_stats[cols_present[0]].abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    plot_data      = summary_stats.loc[top_features, cols_present]
    plot_data_norm = plot_data / plot_data.abs().max()

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_data_norm.plot(
        kind      = "barh",
        ax        = ax,
        color     = [BAR_COLORS[c] for c in cols_present],
        edgecolor = "white",
        width     = 0.7,
    )
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} features by consensus — {axis_name}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(title="Method", fontsize=9, frameon=True,
              framealpha=0.9, edgecolor="grey")
    _apply_bar_theme(ax, xlabel="Normalised importance")
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    plt.show()
    plt.close(fig)


def plot_loadings_heatmap(loadings, axis, top_n=TOP_N, save_name=None):
    """Simple heatmap of the top-N PCA loadings (exploratory version)."""
    top_features = (
        loadings.abs().sum(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        loadings.loc[top_features],
        cmap="coolwarm", center=0, ax=ax,
        linewidths=0.3, cbar_kws={"label": "Loading"},
    )
    ax.set_title(f"Top {top_n} PCA loadings — {axis}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Principal components", fontsize=10)
    ax.set_ylabel("Features", fontsize=10)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    plt.show()
    plt.close(fig)


def plot_loadings_heatmap_pub(loadings, axis_name, top_n=TOP_N, save_name=None):
    """Publication-quality heatmap with annotations."""
    top_features = (
        loadings.abs().sum(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        loadings.loc[top_features],
        cmap="vlag", center=0, ax=ax,
        annot=True, fmt=".2f",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Loading", "shrink": 0.8},
    )
    ax.set_title(f"Top {top_n} PCA loadings — {axis_name}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Principal components", fontsize=10)
    ax.set_ylabel("Features", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    plt.show()
    plt.close(fig)


def plot_cumulative_variance(explained_variance, axis_name="Axis",
                             threshold=0.80, save_name=None):
    """Scree-style plot of cumulative explained variance."""
    cumulative = np.cumsum(explained_variance)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative,
            marker="o", color="#2166AC", lw=1.5, ms=5)
    ax.axhline(threshold, color="#B2182B", ls="--", lw=1.2,
               label=f"{int(threshold * 100)} % variance")
    ax.set_xlabel("Number of PCs", fontsize=11)
    ax.set_ylabel("Cumulative explained variance", fontsize=11)
    ax.set_title(f"Cumulative variance explained — {axis_name}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="grey", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("grey")
    ax.tick_params(labelsize=10, colors="dimgrey")
    ax.grid(color="gainsboro", lw=0.5)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    plt.show()
    plt.close(fig)

# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

for axis in AXES:
    print("\n" + "=" * 60)
    print(f"AXIS: {axis}")
    print("=" * 60)

    data = load_axis_results(axis)

    # Top Elastic Net PCs
    if data["pc_coefficients"] is not None:
        pcs = data["pc_coefficients"].squeeze().abs().sort_values(ascending=False)
        plot_top_series(pcs, f"Top Elastic Net PCs — {axis}",
                        save_name=f"{axis}_EN_top_pcs")
        print(f"  Top PC: {pcs.index[0]}")

    # Top original features (Elastic Net) — absolute and diverging
    feat_infl = None
    if data["feature_influence"] is not None:
        feat_infl = data["feature_influence"].squeeze().sort_values(ascending=False)
        plot_top_series(feat_infl.abs(), f"Top original features (EN) — {axis}",
                        save_name=f"{axis}_EN_top_features_abs")
        plot_diverging_influence(
            feat_infl,
            f"Feature influence — positive vs negative ({axis})",
            save_name=f"{axis}_EN_diverging_influence"
        )

    # Top PCs (Random Forest)
    if data["rf_pc_importance"] is not None:
        rf_pcs = data["rf_pc_importance"].squeeze().sort_values(ascending=False)
        plot_top_series(rf_pcs, f"Top PCs (RF) — {axis}",
                        save_name=f"{axis}_RF_top_pcs")

    # Top original features (Random Forest)
    rf_feat = None
    if data["rf_feature_importance"] is not None:
        rf_feat = data["rf_feature_importance"].squeeze().sort_values(ascending=False)
        plot_top_series(rf_feat, f"Top original features (RF) — {axis}",
                        save_name=f"{axis}_RF_top_features")

    # Consensus feature score
    if feat_infl is not None and rf_feat is not None:
        consensus = compute_consensus(feat_infl, rf_feat)
        plot_top_series(consensus, f"Consensus feature importance — {axis}",
                        save_name=f"{axis}_consensus_importance")

    # PCA loadings heatmaps
    if data["pca_loadings"] is not None:
        plot_loadings_heatmap(data["pca_loadings"], axis,
                              save_name=f"{axis}_pca_loadings_exploratory")
        plot_loadings_heatmap_pub(data["pca_loadings"], axis,
                                  save_name=f"{axis}_pca_loadings_pub")

    # Metrics printout
    if data["metrics"] is not None:
        print("\n  Elastic Net metrics:")
        print(data["metrics"])
    if data["rf_metrics"] is not None:
        print("\n  Random Forest metrics:")
        print(data["rf_metrics"])

    # Summary statistics
    summary_stats = compute_summary_stats(data)
    print(f"\n  Top {TOP_N} features (summary stats):")
    print(summary_stats.head(TOP_N))
    summary_stats.to_csv(RESULTS / f"{axis}_feature_summary.csv")

    # Publication consensus figure
    plot_top_features_pub(summary_stats, axis,
                          save_name=f"{axis}_top_features_pub")

print("\nAll figures saved to:", FIGURES)