library(synthpop)
library(here)
library(ggplot2)
library(reshape2)


# SECTION 1: RELOAD ORIGINAL AND SYNTHETIC DATA

# Adapt the paths below to match your file structure.
# The logic: for each block, load the original and its synthetic counterpart.

# Example for neuroTests blocks — repeat the same pattern for
# neurologicalData blocks and psychometrics.

original_blocks <- list(
  ExecutiveFun     = read.csv(here("CSV", "neuroTests data age residualized", "exec_age_residualized.csv")),
  CognitiveFlex    = read.csv(here("CSV", "neuroTests data age residualized", "cogFlex_age_residualized.csv")),
  conflictAccuracy = read.csv(here("CSV", "neuroTests data age residualized", "conflictAcc_age_residualized.csv")),
  conflictSpe      = read.csv(here("CSV", "neuroTests data age residualized", "conflictSpe_age_residualized.csv"))
)

synthetic_blocks <- list(
  ExecutiveFun     = read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_ExecutiveFun.csv")),
  CognitiveFlex    = read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_CognitiveFlex.csv")),
  conflictAccuracy = read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_conflictAccuracy.csv")),
  conflictSpe      = read.csv(here("synthetic_data", "synthetic neuroTests", "synthetic_conflictSpe.csv"))
)


# SECTION 2: DISTRIBUTIONAL COMPARISON VIA pMSE

# pMSE (propensity score MSE) is the standard utility metric for synthpop.
# Values close to 0 indicate high utility (synthetic closely mirrors original).
# The ratio pMSE / null_pMSE ideally sits near 1.

pmse_results <- list()

for (block_name in names(original_blocks)) {
  
  cat("\nComputing utility for block:", block_name, "\n")
  
  orig <- original_blocks[[block_name]]
  syn  <- synthetic_blocks[[block_name]]
  
  # Align columns
  shared_cols <- intersect(names(orig), names(syn))
  orig <- orig[, shared_cols, drop = FALSE]
  syn  <- syn[, shared_cols, drop = FALSE]
  
  # Reconstruct synthpop object
  syn_obj       <- list()
  syn_obj$syn   <- syn
  syn_obj$m     <- 1
  syn_obj$call  <- match.call()
  class(syn_obj) <- "synds"
  
  # utility.gen is appropriate for continuous data
  utility <- utility.gen(syn_obj, orig, 
                         method = "cart",
                         print.flag = TRUE)
  
  pmse_results[[block_name]] <- utility
  
  cat("Done:", block_name, "\n")
}

for (block_name in names(pmse_results)) {
  cat("\n---", block_name, "---\n")
  print(pmse_results[[block_name]]$pMSE)
}


# SECTION 3: VISUAL DISTRIBUTIONAL COMPARISON

plot_dir <- "validation_plots"
dir.create(plot_dir, showWarnings = FALSE)

for (block_name in names(original_blocks)) {
  
  orig <- original_blocks[[block_name]]
  syn  <- synthetic_blocks[[block_name]]
  shared_cols <- intersect(names(orig), names(syn))
  
  pdf(file.path(plot_dir, paste0(block_name, "_distributions.pdf")),
      width = 10, height = 4)
  
  for (var in shared_cols) {
    df_plot <- rbind(
      data.frame(value = orig[[var]], source = "Original"),
      data.frame(value = syn[[var]],  source = "Synthetic")
    )
    p <- ggplot(df_plot, aes(x = value, fill = source)) +
      geom_density(alpha = 0.5) +
      scale_fill_manual(values = c("Original" = "#2C7BB6", 
                                   "Synthetic" = "#D7191C")) +
      labs(title = paste(block_name, "—", var),
           x = var, y = "Density", fill = "Source") +
      theme_minimal()
    print(p)
  }
  dev.off()
  cat("Saved:", block_name, "\n")
}

# SECTION 4: CORRELATION STRUCTURE PRESERVATION

corr_results <- data.frame(
  block         = character(),
  mean_abs_diff = numeric(),
  stringsAsFactors = FALSE
)

for (block_name in names(original_blocks)) {
  
  orig <- original_blocks[[block_name]]
  syn  <- synthetic_blocks[[block_name]]
  shared_cols <- intersect(names(orig), names(syn))
  
  orig_num <- orig[, shared_cols][, sapply(orig[, shared_cols], is.numeric), drop = FALSE]
  syn_num  <- syn[, shared_cols][, sapply(syn[, shared_cols],  is.numeric), drop = FALSE]
  
  cor_orig <- cor(orig_num, use = "pairwise.complete.obs")
  cor_syn  <- cor(syn_num,  use = "pairwise.complete.obs")
  
  diff_matrix <- cor_orig - cor_syn
  mean_abs    <- mean(abs(diff_matrix[lower.tri(diff_matrix)]))
  
  corr_results <- rbind(corr_results, data.frame(
    block         = block_name,
    mean_abs_diff = round(mean_abs, 4)
  ))
  
  cat("\n---", block_name, "---\n")
  cat("Mean absolute correlation difference:", round(mean_abs, 4), "\n")
}


# SECTION 5: GROUP STRUCTURE PRESERVATION

# Since diagnostic groups are the backbone of the analysis,
# we check that the synthetic behavioral data preserves
# group-level means and standard deviations.

cat("\n--- Group Structure Preservation ---\n")

# Load original and synthetic behavioral files
orig_behavioral <- read.csv(here("final_behavioral.csv"))
syn_behavioral  <- read.csv(here("synthetic_behavioral.csv"))

# Filter out excluded groups to match your analysis subset
orig_behavioral <- orig_behavioral[!(orig_behavioral$group_cumulative %in%
                                       c("DSA", "epilessia")), ]

# Summarize group means and SDs for Age
group_summary <- function(df, group_col, value_col, label) {
  result <- aggregate(df[[value_col]] ~ df[[group_col]],
                      FUN = function(x) c(mean = mean(x), sd = sd(x)))
  result$source <- label
  names(result)[1:2] <- c("group", "stats")
  result
}

orig_summary <- group_summary(orig_behavioral, "group_cumulative", "Age_in_months", "Original")
syn_summary  <- group_summary(syn_behavioral,  "group",            "Age",           "Synthetic")

cat("\nOriginal group age summary:\n")
print(orig_summary)

cat("\nSynthetic group age summary:\n")
print(syn_summary)


# SECTION 6: SUMMARY REPORT

cat("\n========================================\n")
cat("SYNTHPOP VALIDATION SUMMARY\n")
cat("========================================\n")
cat("\nCorrelation structure preservation (lower = better):\n")
print(corr_results)
cat("\nDensity plots saved to:", plot_dir, "\n")
cat("Validation complete.\n")