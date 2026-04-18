# neuroTest_sweeper.R
# Runs the Hoegn-index design-matrix sweep for the neuropsychological test blocks.
# Prerequisite: run DIABLO_designMatrix_neuroTests.R first so that
# `design_matrix` is available in the environment.
#
# IMPORTANT: set ncomp consistently throughout this script.
#            Then activate ONLY the matching metric-extraction loop:
#              - ncomp = 1  → use the "NCOMP = 1" loop (lines ~285)
#              - ncomp > 1  → use the "NCOMP > 1" loop (lines ~303)
#            Comment out the loop you are NOT using.

library(mixOmics)
library(vegan)
library(dplyr)
library(here)
library(stats)

# ----------------------------
# Load outcome variable
psychometrics <- read.csv(here("synthetic_behavioral.csv"))
psychometrics <- psychometrics[!(psychometrics$group %in% c("DSA", "epilessia")), ]

# Load neuropsychological test blocks
exec        <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ExecutiveFun.csv"))
cogFlex     <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_CognitiveFlex.csv"))
conflictAcc <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ConflictAccuracy.csv"))
conflictSpe <- read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ConflictSpe.csv"))

###############################################################################
# DIABLO PARAMETER DEFINITION
###############################################################################

# Build block list (converted to matrices for mixOmics compatibility)
X <- list(
  ExecutiveFun     = as.matrix(exec),
  CognitiveFlex    = as.matrix(cogFlex),
  conflictAccuracy = as.matrix(conflictAcc),
  conflictSpe      = as.matrix(conflictSpe)
)

# Outcome vector
Y <- psychometrics$group

# Keep all features per block (no sparsity on the sweep)
keepX <- lapply(X, ncol)

# Sanity check: subject counts must match across all blocks and Y
sapply(X, function(x) dim(x))  # [1,] should be identical across columns
length(Y)

###############################################################################
# GLOBAL EMBEDDING FUNCTIONS
###############################################################################

# Average variates across blocks → single subject embedding (ncomp = 1 only)
global_embedding_n1 <- function(diablo_obj) {
  Z <- Reduce("+", diablo_obj$variates) / length(diablo_obj$variates)
  scale(Z)
}

# Two-option embedding for ncomp > 1:
#   $flat  — concatenate all block components (subjects x [ncomp * n_blocks])
#   $avg   — average across blocks per component (subjects x ncomp)
global_embedding_multi <- function(diablo_obj) {
  ncomp <- ncol(diablo_obj$variates[[1]])
  if (ncomp == 1) {
    return(global_embedding_n1(diablo_obj))
  }
  
  # Option 1: flatten all block components
  Z_flat <- do.call(cbind, diablo_obj$variates)
  Z_flat <- scale(Z_flat)
  
  # Option 2: average across blocks per component (subjects x ncomp)
  Z_avg <- Reduce("+", lapply(diablo_obj$variates, function(x) x[, 1:ncomp])) /
    length(diablo_obj$variates)
  Z_avg <- scale(Z_avg)
  
  list(flat = Z_flat, avg = Z_avg)
}

###############################################################################
# REFERENCE DIABLO MODEL (used as Procrustes anchor)
###############################################################################

# `design_matrix` must be loaded from DIABLO_designMatrix_neuroTests.R
design_matrix_ref <- design_matrix

diablo_ref <- block.splsda(
  X      = X,
  Y      = Y,
  ncomp  = 4,        # IMPORTANT: keep consistent with the sweep loop below
  keepX  = keepX,
  design = design_matrix_ref
)

# Select the embedding function that matches ncomp:
#   global_embedding_n1    for ncomp = 1
#   global_embedding_multi for ncomp > 1
Z_ref <- global_embedding_multi(diablo_ref)

###############################################################################
# HOEGN-INDEX INTERPOLATION
###############################################################################

# Interpolates design matrices along the logit-transformed path from M_start
# (zero matrix) to M_end (all-ones matrix), placing M_ref proportionally
# on that path via least-squares projection.
hoegn_interpolation <- function(M_start, M_ref, n_steps = 300) {
  
  stopifnot(all(dim(M_start) == dim(M_ref)))
  
  n     <- nrow(M_start)
  M_end <- matrix(1, n, n,
                  dimnames = list(rownames(M_ref), colnames(M_ref)))
  
  # Logit / inverse-logit helpers (with epsilon guard against boundary values)
  logit     <- function(x) log((x + 1e-6) / (1 - x + 1e-6))
  inv_logit <- function(x) exp(x) / (1 + exp(x))
  
  # Vectorise and transform all three matrices
  vec_start_logit <- logit(as.vector(M_start))
  vec_ref_logit   <- logit(as.vector(M_ref))
  vec_end_logit   <- logit(as.vector(M_end))
  
  # Least-squares projection of M_ref onto the start→end line
  numerator   <- sum((vec_ref_logit - vec_start_logit) * (vec_end_logit - vec_start_logit))
  denominator <- sum((vec_end_logit - vec_start_logit)^2)
  alpha_ref   <- numerator / denominator
  
  # Clamp to [0, 1] in case of floating-point overshoot
  alpha_ref <- max(min(alpha_ref, 1), 0)
  
  # Build alpha grid with denser sampling around alpha_ref
  n_before   <- ceiling(n_steps * alpha_ref)
  n_after    <- n_steps - n_before
  alpha_grid <- c(
    seq(0,         alpha_ref, length.out = max(n_before, 1)),
    seq(alpha_ref, 1,         length.out = max(n_after,  1))
  )
  
  # Interpolate one design matrix per alpha value
  design_grid <- lapply(alpha_grid, function(a) {
    vec_current_logit <- (1 - a) * vec_start_logit + a * vec_end_logit
    vec_current       <- inv_logit(vec_current_logit)
    matrix(vec_current, n, n,
           dimnames = list(rownames(M_ref), colnames(M_ref)))
  })
  
  list(
    alpha_grid  = alpha_grid,
    design_grid = design_grid,
    alpha_ref   = alpha_ref
  )
}

# Build zero-matrix starting point with matching block names
n_blocks <- nrow(design_matrix)
M_start  <- matrix(0, n_blocks, n_blocks,
                   dimnames = list(rownames(design_matrix), colnames(design_matrix)))

n_steps <- 300
result  <- hoegn_interpolation(M_start, design_matrix, n_steps = n_steps)

design_grid <- result$design_grid
alpha_grid  <- result$alpha_grid
alpha_ref   <- result$alpha_ref

# Save interpolation metadata for later design-matrix extrapolation
interp_result <- list(
  M_start    = M_start,
  M_ref      = design_matrix,
  alpha_grid = alpha_grid,
  alpha_ref  = alpha_ref
)
saveRDS(interp_result, "hoegn_interpolation_result.rds")

###############################################################################
# PRE-SWEEP INITIALISATION
###############################################################################

# Container for all fitted DIABLO objects
diablo_results <- vector("list", length(design_grid))

# Results table: one row per alpha step
results_df <- data.frame(
  alpha      = numeric(n_steps),
  trust      = numeric(n_steps),
  continuity = numeric(n_steps),
  shepard    = numeric(n_steps),
  procrustes = numeric(n_steps)
)

# Precompute high-dimensional distance matrix (used by all quality metrics)
X_concat <- scale(do.call(cbind, X))  # subjects x all_features
D_high   <- dist(X_concat)

###############################################################################
# QUALITY METRIC FUNCTIONS
###############################################################################

# Continuity: fraction of true high-dim neighbours preserved in low-dim space
continuity_manual <- function(X, Z, k = 10) {
  n  <- nrow(X)
  Dx <- as.matrix(dist(X))
  Dz <- as.matrix(dist(Z))
  Rx <- apply(Dx, 1, rank)
  Rz <- apply(Dz, 1, rank)
  
  c_sum <- 0
  for (i in 1:n) {
    # V_k(i): points that are close in X but far in Z
    V     <- which(Rx[i, ] <= k & Rz[i, ] > k)
    c_sum <- c_sum + sum(Rz[i, V] - k)
  }
  1 - (2 / (n * k * (2 * n - 3 * k - 1))) * c_sum
}

# Trustworthiness: fraction of apparent low-dim neighbours that are true neighbours
trustworthiness_manual <- function(X, Z, k = 10) {
  n  <- nrow(X)
  Dx <- as.matrix(dist(X))
  Dz <- as.matrix(dist(Z))
  Rx <- apply(Dx, 1, rank)
  Rz <- apply(Dz, 1, rank)
  
  tw_sum <- 0
  for (i in 1:n) {
    # U_k(i): points that appear close in Z but are far in X
    U <- which(Rz[i, ] <= k & Rx[i, ] > k)
    if (length(U) > 0) {
      tw_sum <- tw_sum + sum(Rx[i, U] - k)
    }
  }
  1 - (2 / (n * k * (2 * n - 3 * k - 1))) * tw_sum
}

###############################################################################
# DESIGN MATRIX SWEEP
###############################################################################

for (i in seq_along(design_grid)) {
  
  design_i <- design_grid[[i]]
  
  diablo_i <- block.splsda(
    X      = X,
    Y      = Y,
    ncomp  = 4,        # IMPORTANT: must match ncomp used for diablo_ref above
    keepX  = keepX,
    design = design_i
  )
  
  diablo_results[[i]] <- diablo_i
  results_df$alpha[i] <- alpha_grid[i]
}

###############################################################################
# METRIC EXTRACTION — NCOMP = 1
# Run this block only when ncomp = 1; comment it out otherwise.
###############################################################################

for (i in seq_along(diablo_results)) {
  Z_i <- global_embedding_n1(diablo_results[[i]])
  
  results_df$trust[i]      <- trustworthiness_manual(X_concat, Z_i, k = 10)
  results_df$continuity[i] <- continuity_manual(X_concat, Z_i, k = 10)
  results_df$shepard[i]    <- cor(as.vector(D_high), as.vector(dist(Z_i)), method = "spearman")
  results_df$procrustes[i] <- protest(Z_ref, Z_i, permutations = 0)$t0
  results_df$procrustes[i] <- 1 - results_df$procrustes[i]
}

Z_list <- lapply(diablo_results, global_embedding_n1)

###############################################################################
# METRIC EXTRACTION — NCOMP > 1
# Run this block only when ncomp > 1; comment it out otherwise.
# Uses Z_flat for trust/continuity (full dimensionality) and
# Z_avg for Shepard/Procrustes (averaged across blocks).
###############################################################################

# Convert the reference embedding list to a single scaled matrix for Procrustes
Z_ref_mat           <- do.call(cbind, Z_ref)
Z_ref_mat           <- scale(Z_ref_mat)
rownames(Z_ref_mat) <- rownames(X_concat)

for (i in seq_along(diablo_results)) {
  Z_i_list <- global_embedding_multi(diablo_results[[i]])
  
  Z_flat <- as.matrix(Z_i_list$flat)
  Z_avg  <- as.matrix(Z_i_list$avg)
  
  rownames(Z_flat) <- rownames(X_concat)
  rownames(Z_avg)  <- rownames(X_concat)
  
  results_df$trust[i]      <- trustworthiness_manual(X_concat, Z_flat, k = 10)
  results_df$continuity[i] <- continuity_manual(X_concat, Z_flat, k = 10)
  results_df$shepard[i]    <- cor(as.vector(D_high), as.vector(dist(Z_avg)), method = "spearman")
  results_df$procrustes[i] <- vegan::procrustes(Z_ref_mat, Z_avg)$ss
  results_df$procrustes[i] <- 1 - results_df$procrustes[i]
}

###############################################################################
# SMOOTHED METRIC CURVES
###############################################################################

# Fits a smoothing spline to a metric vector over alpha
# Returns smoothed values at each alpha point
plateau_metrics <- function(alpha, metric_values, spar = 0.8) {
  smooth_metric <- smooth.spline(alpha, metric_values, spar = spar)
  predict(smooth_metric, alpha)$y
}

trust_metrics      <- plateau_metrics(results_df$alpha, results_df$trust)
continuity_metrics <- plateau_metrics(results_df$alpha, results_df$continuity)
shepard_metrics    <- plateau_metrics(results_df$alpha, results_df$shepard)

# Procrustes is min-max normalised before smoothing (its scale differs from others)
procrustes_peak_normalized <- function(alpha, metric_values, spar = 0.7) {
  d_min              <- min(metric_values)
  d_max              <- max(metric_values)
  metric_values_norm <- (metric_values - d_min) / (d_max - d_min)
  
  smooth_metric <- smooth.spline(alpha, metric_values_norm, spar = spar)
  metric_smooth <- predict(smooth_metric, alpha)$y
  
  # Clamp to [0, 1] after smoothing
  metric_smooth[metric_smooth < 0] <- 0
  metric_smooth[metric_smooth > 1] <- 1
  
  metric_smooth
}

procrustes_metrics <- procrustes_peak_normalized(
  results_df$alpha,
  results_df$procrustes,
  spar = 0.75
)