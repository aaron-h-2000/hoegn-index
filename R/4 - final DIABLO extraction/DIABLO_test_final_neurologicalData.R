# DIABLO_test_final_neurologicalData.R
# Fits the final DIABLO model for the neurological blocks using the optimal
# design matrix (design_matrix_star) recovered by the Hoegn Index extrapolator,
# then exports the single-axis global subject embedding for downstream UMAP.
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

# Load neurological data blocks
coh  <- read.csv(here("synthetic_data", "synthetic neurologicalData", "synthetic_Coh.csv"))
exc  <- read.csv(here("synthetic_data", "synthetic neurologicalData", "synthetic_Exc.csv"))
ap   <- read.csv(here("synthetic_data", "synthetic neurologicalData", "synthetic_Ap.csv"))
flex <- read.csv(here("synthetic_data", "synthetic neurologicalData", "synthetic_Flex.csv"))
flu  <- read.csv(here("synthetic_data", "synthetic neurologicalData", "synthetic_Flu.csv"))

# Build block list
X <- list(
  Coh  = as.matrix(coh),
  Exc  = as.matrix(exc),
  Ap   = as.matrix(ap),
  Flex = as.matrix(flex),
  Flu  = as.matrix(flu)
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

write.csv(Z, "neurologicalData_1axis_noPy.csv", row.names = FALSE)