# plots.py
# Generates the 3-D scatter plot of the joint manifold (Age x
# neurological axis x neuroTests axis) coloured by diagnostic group,
# and exports it as interactive HTML and four static PNGs (one per
# canonical camera angle).

from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT    = Path(r"C:\Users\aaron\Documents\Python\embedding paper")
OUT_DIR = ROOT / "results" / "figures" / "manifold"
OUT_DIR.mkdir(parents = True, exist_ok = True)

# =============================================================================
# LOAD AND RECODE
# =============================================================================

df = pd.read_csv(
    ROOT / "results" / "final embedding" /
    "neurologicalData_PHATE_plus_neuroTests_UMAP_manifold.csv"
)

df["group"] = df["group"].map({
    "controlli": "Controls",
    "ADHD":      "ADHD",
    "ASD":       "ASD",
})

# =============================================================================
# COLOUR PALETTE
# =============================================================================

GROUP_COLORS = {
    "Controls": "#4393C3",   # blue — brighter on dark bg
    "ADHD":     "#F4A442",   # amber
    "ASD":      "#E06C6C",   # coral-red
}

# =============================================================================
# BASE FIGURE
# =============================================================================

fig = px.scatter_3d(
    df,
    x     = "Age",
    y     = "neurologicalData",
    z     = "neuroTests",
    color = "group",
    color_discrete_map = GROUP_COLORS,
    title = "Joint manifold: PHATE neurological x UMAP neuroTests axes",
    labels = {
        "Age":              "Age (months)",
        "neurologicalData": "Neurological component (PHATE)",
        "neuroTests":       "Behavioural component (UMAP)",
        "group":            "Group",
    },
    opacity = 0.92,
)

# Larger dots, white border for separation
fig.update_traces(
    marker = dict(
        size = 7,
        line = dict(width = 0.5, color = "white"),
    )
)

# Dark background layout
fig.update_layout(
    scene = dict(
        xaxis_title = "Age (months)",
        yaxis_title = "Neurological component (PHATE)",
        zaxis_title = "Behavioural component (UMAP)",
        bgcolor     = "rgb(20, 20, 30)",
        xaxis = dict(
            backgroundcolor = "rgb(30, 30, 45)",
            gridcolor       = "rgba(255,255,255,0.15)",
            showbackground  = True,
            zerolinecolor   = "rgba(255,255,255,0.3)",
            tickfont        = dict(color = "white"),
            title_font      = dict(color = "white"),
        ),
        yaxis = dict(
            backgroundcolor = "rgb(25, 25, 40)",
            gridcolor       = "rgba(255,255,255,0.15)",
            showbackground  = True,
            zerolinecolor   = "rgba(255,255,255,0.3)",
            tickfont        = dict(color = "white"),
            title_font      = dict(color = "white"),
        ),
        zaxis = dict(
            backgroundcolor = "rgb(20, 20, 35)",
            gridcolor       = "rgba(255,255,255,0.15)",
            showbackground  = True,
            zerolinecolor   = "rgba(255,255,255,0.3)",
            tickfont        = dict(color = "white"),
            title_font      = dict(color = "white"),
        ),
    ),
    paper_bgcolor     = "rgb(15, 15, 25)",
    plot_bgcolor      = "rgb(15, 15, 25)",
    font              = dict(family = "Arial", size = 12, color = "white"),
    legend_title_text = "Diagnostic group",
    legend            = dict(
        font        = dict(color = "white"),
        bgcolor     = "rgba(255,255,255,0.05)",
        bordercolor = "rgba(255,255,255,0.2)",
        borderwidth = 1,
    ),
    title_font = dict(color = "white", size = 14),
    margin     = dict(l = 0, r = 0, t = 50, b = 0),
)

# =============================================================================
# FOUR CANONICAL CAMERA ANGLES
# =============================================================================

CAMERAS = {
    "diagonal":  dict(eye = dict(x = 1.5,  y = 1.5,  z = 1.0)),   # main overview
    "top_down":  dict(eye = dict(x = 0.0,  y = 0.0,  z = 2.5)),   # neuro x behaviour, no age
    "front":     dict(eye = dict(x = 0.0,  y = 2.5,  z = 0.5)),   # behaviour x age
    "side":      dict(eye = dict(x = 2.5,  y = 0.0,  z = 0.5)),   # neuro x age
}

for angle_name, camera in CAMERAS.items():
    fig.update_layout(scene_camera = camera)
    out_path = OUT_DIR / f"manifold_{angle_name}.png"
    fig.write_image(str(out_path), width = 1400, height = 1000, scale = 2)
    print(f"Saved: {out_path}")

# =============================================================================
# INTERACTIVE HTML (uses last camera angle — reset to diagonal)
# =============================================================================

fig.update_layout(scene_camera = CAMERAS["diagonal"])
html_path = OUT_DIR / "neurologicalData_PHATE_plus_neuroTests_UMAP_manifold.html"
fig.write_html(str(html_path))
print(f"Saved: {html_path}")

fig.show(renderer="browser")
print("\nAll manifold figures saved to:", OUT_DIR)