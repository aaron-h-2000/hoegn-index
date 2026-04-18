# dataframe_joining_for_manifold.py
# Joins the behavioral metadata with the UMAP embedding axes from both
# pipelines into a single analysis-ready dataframe.
#
# NOTE: The join is purely positional (pd.concat along axis = 1), so the row
# order of the UMAP CSVs must match the row order of the behavioral CSV
# after group filtering. Verify alignment before downstream analyses.

from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\aaron\Documents\Python\embedding paper")

# =============================================================================
# LOAD DATA
# =============================================================================

df_behavioral = pd.read_csv(ROOT / "synthetic_data" / "synthetic_behavioral.csv")

# UMAP embedding axes (one column each)
df_neuro = pd.read_csv(ROOT / "results" / "final axes" / "neurologicalData_1axis_PHATE.csv")
df_tests = pd.read_csv(ROOT / "results" / "final axes" / "neuroTests_1axis_UMAP.csv")

# =============================================================================
# RENAME AND JOIN
# =============================================================================

df_neuro = df_neuro.rename(columns={"PHATE_comp1": "neurologicalData"})
df_tests = df_tests.rename(columns={"UMAP_comp1": "neuroTests"})

# Positional concatenation — assumes rows are already aligned across all CSVs
df_final = pd.concat(
    [
        df_behavioral.reset_index(drop=True),
        df_neuro[["neurologicalData"]].reset_index(drop = True),
        df_tests[["neuroTests"]].reset_index(drop = True),
    ],
    axis=1,
)

# Drop the pandas artefact column if present (errors="ignore" avoids a crash
# when the column is absent)
df_final.drop(columns = ["Unnamed: 0"], inplace = True, errors = "ignore")

# =============================================================================
# SANITY CHECK AND SAVE
# =============================================================================

print("Shape:", df_final.shape)
print("Columns:", df_final.columns.tolist())
print(df_final.head())

df_final.to_csv(
    ROOT / "neurologicalData_PHATE_plus_neuroTests_UMAP_manifold.csv",
    index=False,
)