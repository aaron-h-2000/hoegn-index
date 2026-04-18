# DIABLO_test_final_neuroTests.R
# Fits the final DIABLO model for the neuropsychological test blocks using the
# optimal design matrix (design_matrix_star) recovered by the Hoegn Index
# extrapolator, then exports the single-axis global subject embedding for
# downstream UMAP.
#
# Prerequisites:
#   - Hoegn_index_design_matrix_extrapolator.R must have been run so that
#     `design_matrix_star` is available in the environment (or loaded via
#     readRDS("design_matrix_star.rds")).

library(mixOmics)
library(dplyr)
library(here)

# ----------------------------
# Load outcome variable
psychometrics <- read.csv(here("synthetic_behavioral.csv"))
psychometrics <- psychometrics[!(psychometrics$group %in% c("DSA", "epilessia")), ]

# Load neuropsychological test blocks
exec        <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ExecutiveFun.csv"))
cogFlex     <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_CognitiveFlex.csv"))
conflictAcc <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ConflictAccuracy.csv"))
conflictSpe <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ConflictSpe.csv"))

# Build block list
X <- list(
  ExecutiveFun     = as.matrix(exec),
  CognitiveFlex    = as.matrix(cogFlex),
  conflictAccuracy = as.matrix(conflictAcc),
  conflictSpe      = as.matrix(conflictSpe)
)

# Outcome vector
Y <- psychometrics$group

# Sanity check: subject counts must match across all blocks and Y
sapply(X, function(x) dim(x))  # [1,] should be identical across columns
length(Y)

keepX <- lapply(X, ncol)

# ----------------------------
# Fit final DIABLO model with the Hoegn-optimal design matrix
# (design_matrix_star is produced by Hoegn_index_design_matrix_extrapolator.R)

diablo <- block.splsda(
  X      = X,
  Y      = Y,
  ncomp  = 1,
  keepX  = keepX,
  design = design_matrix_star
)

# ----------------------------
# Compute global subject embedding (average block scores)

scores <- diablo$variates
Z      <- Reduce("+", scores) / length(scores)

write.csv(Z, "neuroTests_1axis_noPy.csv", row.names = FALSE)