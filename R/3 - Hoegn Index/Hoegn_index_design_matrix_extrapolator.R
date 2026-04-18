# Hoegn_index_design_matrix_extrapolator.R
# Recovers the optimal design matrix corresponding to x_star (the Hoegn Index
# optimum).  x_star is on the [0, 10] display scale used in Hoegn_Index.R;
# it is converted back to alpha in [0, 1] before re-running the logit
# interpolation (same arithmetic as inside hoegn_interpolation()).

library(here)

# ----------------------------
# Load saved results from previous scripts

index_result  <- readRDS(here("hoegn_index_result.rds"))
interp_result <- readRDS(here("hoegn_interpolation_result.rds"))

# ----------------------------
# Extract key values

x_star  <- index_result$x_star   # optimal x on [0, 10] display scale
M_start <- interp_result$M_start  # zero-matrix (interpolation start)
M_ref   <- interp_result$M_ref    # data-driven reference design matrix

n <- nrow(M_start)

# Convert x_star from display scale [0, 10] back to alpha [0, 1]
alpha_star <- x_star / 10

# ----------------------------
# Logit / inverse-logit helpers (same as in the sweeper)

logit     <- function(x) log((x + 1e-6) / (1 - x + 1e-6))
inv_logit <- function(x) exp(x) / (1 + exp(x))

# Build all-ones endpoint matrix
M_end <- matrix(1, n, n,
                dimnames = list(rownames(M_ref), colnames(M_ref)))

# ----------------------------
# Reconstruct the optimal design matrix at alpha_star

vec_start_logit <- logit(as.vector(M_start))
vec_end_logit   <- logit(as.vector(M_end))

design_matrix_star <- matrix(
  inv_logit((1 - alpha_star) * vec_start_logit + alpha_star * vec_end_logit),
  n, n,
  dimnames = list(rownames(M_ref), colnames(M_ref))
)

# ----------------------------
# Save and display

saveRDS(design_matrix_star, "design_matrix_star.rds")
print(design_matrix_star)