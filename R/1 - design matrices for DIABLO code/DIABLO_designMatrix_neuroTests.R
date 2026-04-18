# DIABLO_designMatrix_neuroTests.R
# Computes the inter-block design matrix for the neuropsychological test blocks,
# using mean absolute pairwise correlation. The resulting design_matrix is used
# as input to the DIABLO sweeper script.

library(here)

# Load individual data blocks
exec        <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ExecutiveFun.csv"))
cogFlex     <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_CognitiveFlex.csv"))
conflictAcc <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ConflictAccuracy.csv"))
conflictSpe <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ConflictSpe.csv"))

# Collect blocks into a named list
blocks <- list(
  ExecutiveFun     = exec,
  CognitiveFlex    = cogFlex,
  conflictAccuracy = conflictAcc,
  conflictSpe      = conflictSpe
)

# Sanity check: row counts (subjects) must match across all blocks
sapply(blocks, function(x) dim(x))  # [1,] should be identical across columns

# ----------------------------
# Compute inter-block design matrix via mean absolute correlation

n_blocks <- length(blocks)

# Initialize symmetric correlation matrix
cor_mat <- matrix(NA, nrow = n_blocks, ncol = n_blocks)
rownames(cor_mat) <- colnames(cor_mat) <- names(blocks)

for (i in 1:n_blocks) {
  for (j in i:n_blocks) {
    x <- blocks[[i]]  # subjects x features
    y <- blocks[[j]]
    
    # Full correlation matrix between all features of block i and block j
    cor_xy <- cor(x, y)
    
    # Summarise as mean absolute correlation (symmetric, sign-agnostic)
    mean_corr <- mean(abs(cor_xy))
    
    cor_mat[i, j] <- mean_corr
    cor_mat[j, i] <- mean_corr
  }
}

# Round for readability; passed directly to DIABLO as the design matrix
design_matrix <- round(cor_mat, 3)