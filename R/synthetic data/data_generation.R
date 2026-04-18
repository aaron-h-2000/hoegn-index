library(synthpop)
library(here)

set.seed(13)

output_dir <- "synthetic_data"
dir.create(output_dir, showWarnings = FALSE)

################################################################################

psychometrics <- read.csv(here("final_behavioral.csv"))

################################################################################

coh  <- read.csv(here("CSV", "neurologicalData data age residualized", "coh_age_residualized.csv"))
exc  <- read.csv(here("CSV", "neurologicalData data age residualized", "exc_age_residualized.csv"))
ap   <- read.csv(here("CSV", "neurologicalData data age residualized", "ap_age_residualized.csv"))
flex <- read.csv(here("CSV", "neurologicalData data age residualized", "flex_age_residualized.csv"))
flu  <- read.csv(here("CSV", "neurologicalData data age residualized", "flu_age_residualized.csv"))

coh$group_cumulative  <- psychometrics$group_cumulative 
exc$group_cumulative  <- psychometrics$group_cumulative
ap$group_cumulative   <- psychometrics$group_cumulative
flex$group_cumulative <- psychometrics$group_cumulative
flu$group_cumulative  <- psychometrics$group_cumulative

coh  <- coh[!(coh$group_cumulative %in% c("DSA", "epilessia")), ]
exc  <- exc[!(exc$group_cumulative %in% c("DSA", "epilessia")), ]
ap   <- ap[!(ap$group_cumulative %in% c("DSA", "epilessia")), ]
flex <- flex[!(flex$group_cumulative %in% c("DSA", "epilessia")), ]
flu  <- flu[!(flu$group_cumulative %in% c("DSA", "epilessia")), ]

coh$group_cumulative  <- NULL
exc$group_cumulative  <- NULL
ap$group_cumulative   <- NULL
flex$group_cumulative <- NULL
flu$group_cumulative  <- NULL

neurologicalData <- list(
  Coh  = as.matrix(coh),
  Exc  = as.matrix(exc),
  Ap   = as.matrix(ap),
  Flex = as.matrix(flex),
  Flu  = as.matrix(flu)
)

################################################################################

exec        <- read.csv(here("CSV", "neuroTests data age residualized", "exec_age_residualized.csv"))
cogFlex     <- read.csv(here("CSV", "neuroTests data age residualized", "cogFlex_age_residualized.csv"))
conflictAcc <- read.csv(here("CSV", "neuroTests data age residualized", "conflictAcc_age_residualized.csv"))
conflictSpe <- read.csv(here("CSV", "neuroTests data age residualized", "conflictSpe_age_residualized.csv"))

exec$group_cumulative        <- psychometrics$group_cumulative 
cogFlex$group_cumulative     <- psychometrics$group_cumulative
conflictAcc$group_cumulative <- psychometrics$group_cumulative
conflictSpe$group_cumulative <- psychometrics$group_cumulative

exec        <- exec[!(exec$group_cumulative %in% c("DSA", "epilessia")), ]
cogFlex     <- cogFlex[!(cogFlex$group_cumulative %in% c("DSA", "epilessia")), ]
conflictAcc <- conflictAcc[!(conflictAcc$group_cumulative %in% c("DSA", "epilessia")), ]
conflictSpe <- conflictSpe[!(conflictSpe$group_cumulative %in% c("DSA", "epilessia")), ]

exec$group_cumulative        <- NULL 
cogFlex$group_cumulative     <- NULL
conflictAcc$group_cumulative <- NULL
conflictSpe$group_cumulative <- NULL

neuroTests <- list(ExecutiveFun = exec,
                   CognitiveFlex = cogFlex,
                   conflictAccuracy = conflictAcc,
                   conflictSpe = conflictSpe)

################################################################################

psychometrics <- psychometrics[!(psychometrics$group_cumulative %in% c("DSA", "epilessia")), ]

################################################################################

synthetic_blocks <- list()

# the var inside of the for loop in names() and in block_df <- as the index for 
# [[block_name]] is the var to be changed by the lists we are synthesizing

for (block_name in names(neuroTests)) {
  
  cat("\nGenerating synthetic block:", block_name, "\n")
  
  block_df <- as.data.frame(neuroTests[[block_name]])
  
  # synthpop model
  syn_obj <- syn(
    block_df,
    seed = 13,
    print.flag = FALSE
  )
  
  synthetic_blocks[[block_name]] <- syn_obj$syn
  
  cat("Done:", block_name,
      "| shape =", dim(syn_obj$syn)[1],
      "x", dim(syn_obj$syn)[2], "\n")
}

for (block_name in names(synthetic_blocks)) {
  
  write.csv(
    synthetic_blocks[[block_name]],
    file = file.path(
      output_dir,
      paste0("synthetic_", block_name, ".csv")
    ),
    row.names = FALSE
  )
}

cat("CSV export complete.\n")

################################################################################

synthetic_age <- rnorm(
  n = length(psychometrics$Age_in_months),
  mean = mean(psychometrics$Age_in_months),
  sd = sd(psychometrics$Age_in_months)
)

synth_behavioral <- data.frame(Age = synthetic_age, group = psychometrics$group_cumulative)

write.csv(synth_behavioral, "synthetic_behavioral.csv")