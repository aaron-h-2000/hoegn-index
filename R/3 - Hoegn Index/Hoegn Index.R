# Hoegn_Index.R
# Computes the Hoegn Index from the four smoothed quality metrics produced by
# the sweeper scripts. The index is the pointwise minimum (envelope) across all
# metrics, and its maximum identifies the optimal alpha (design-matrix position).
#
# Note: x_grid is scaled to [0, 10] purely for visual convenience on the plot
# x-axis.  The true alpha lives in [0, 1]; conversion is alpha = x / 10.
# alpha_ref (from the sweeper) is therefore multiplied by 10 before plotting.

# ----------------------------
# 1. Collect smoothed metric vectors from the sweeper

f1 <- trust_metrics
f2 <- continuity_metrics
f3 <- shepard_metrics
f4 <- procrustes_metrics

f_list <- list(f1, f2, f3, f4)

n      <- length(f1)
x_grid <- seq(0, 10, length.out = n)  # display scale; true alpha = x / 10

# ----------------------------
# 2. Hoegn Index envelope: pointwise minimum across all metrics

f_env <- do.call(pmin, f_list)

# ----------------------------
# 3. Optimal point: maximum of the envelope

A      <- max(f_env)
x_star <- x_grid[which.max(f_env)]

# ----------------------------
# 4. Confidence / plateau width around the optimum

threshold  <- 0.99 * A
idx        <- which(f_env >= threshold)
x_region   <- x_grid[idx]
uncertainty <- max(x_region) - min(x_region)
confidence  <- 1 - (uncertainty / (max(x_grid) - min(x_grid)))

# ----------------------------
# 5. Per-metric values at the optimal point

idx_star           <- which.max(f_env)
trust_at_star      <- f1[idx_star]
continuity_at_star <- f2[idx_star]
shepard_at_star    <- f3[idx_star]
procrustes_at_star <- f4[idx_star]

# ----------------------------
# 6. Collect and save results

result <- list(
  x_star             = x_star,
  A                  = A,
  uncertainty        = uncertainty,
  confidence         = confidence,
  trust_at_star      = trust_at_star,
  continuity_at_star = continuity_at_star,
  shepard_at_star    = shepard_at_star,
  procrustes_at_star = procrustes_at_star
)

print(result)
saveRDS(result, "hoegn_index_result.rds")

# ----------------------------
# 7. Save full sweep data to CSV
# This is the file you need for the benchmarking comparison table

sweep_df <- data.frame(
  x_grid         = x_grid,
  alpha          = x_grid / 10,       # true alpha in [0, 1]
  trustworthiness = f1,
  continuity      = f2,
  shepard         = f3,
  procrustes      = f4,
  hoegn_index     = f_env
)

write.csv(sweep_df, "hoegn_sweep_metrics_neuroTests_ncomp4.csv", row.names = FALSE)
cat("Full sweep metrics saved to hoegn_sweep_metrics.csv\n")

# ----------------------------
# 8. Single-metric benchmark comparison
# For each metric, find what parameter it alone would have selected
# and record what ALL other metrics look like at that point
# This is the benchmarking table for the paper

single_metric_names <- c("trustworthiness", "continuity", "shepard", "procrustes")
single_metric_vecs  <- list(f1, f2, f3, f4)

benchmark_rows <- lapply(seq_along(single_metric_names), function(i) {
  idx_best <- which.max(single_metric_vecs[[i]])
  data.frame(
    selection_method   = paste0("argmax_", single_metric_names[i]),
    x_selected         = x_grid[idx_best],
    alpha_selected     = x_grid[idx_best] / 10,
    trustworthiness    = f1[idx_best],
    continuity         = f2[idx_best],
    shepard            = f3[idx_best],
    procrustes         = f4[idx_best],
    hoegn_index        = f_env[idx_best]
  )
})

# Add the Hoegn Index row
hoegn_row <- data.frame(
  selection_method   = "hoegn_index",
  x_selected         = x_star,
  alpha_selected     = x_star / 10,
  trustworthiness    = trust_at_star,
  continuity         = continuity_at_star,
  shepard            = shepard_at_star,
  procrustes         = procrustes_at_star,
  hoegn_index        = A
)

benchmark_df <- do.call(rbind, c(benchmark_rows, list(hoegn_row)))

write.csv(benchmark_df, "hoegn_benchmark_comparison_neuroTests_ncomp4.csv", row.names = FALSE)
cat("Benchmark comparison saved to hoegn_benchmark_comparison.csv\n")
print(benchmark_df)

###############################################################################
# PLOT: Hoegn Index — ggplot2 version
###############################################################################

library(ggplot2)
library(tidyr)
library(dplyr)

# ----------------------------
# Assemble long-format data frame for the metric lines

plot_df <- data.frame(
  x             = x_grid,
  Trustworthiness = f1,
  Continuity      = f2,
  Shepard         = f3,
  Procrustes      = f4,
  `Hoegn Index`   = f_env,
  check.names = FALSE
) |>
  pivot_longer(-x, names_to = "Metric", values_to = "value")

# Factor order controls legend order
metric_levels <- c("Trustworthiness", "Continuity", "Shepard",
                   "Procrustes", "Hoegn Index")
plot_df$Metric <- factor(plot_df$Metric, levels = metric_levels)

# ----------------------------
# Colour / linetype / linewidth scales

metric_colours <- c(
  "Trustworthiness" = "#2166AC",   # blue
  "Continuity"      = "#4DAC26",   # green
  "Shepard"         = "#7B2D8B",   # purple
  "Procrustes"      = "#E6851E",   # amber
  "Hoegn Index"     = "#B2182B"    # dark red
)

metric_linetypes <- c(
  "Trustworthiness" = "solid",
  "Continuity"      = "solid",
  "Shepard"         = "solid",
  "Procrustes"      = "solid",
  "Hoegn Index"     = "dashed"
)

metric_linewidths <- c(
  "Trustworthiness" = 0.8,
  "Continuity"      = 0.8,
  "Shepard"         = 0.8,
  "Procrustes"      = 0.8,
  "Hoegn Index"     = 1.3
)

# ----------------------------
# Confidence region data (only drawn if the plateau region has width)

conf_rect <- if (length(x_region) > 1) {
  data.frame(xmin = min(x_region), xmax = max(x_region),
             ymin = -Inf,          ymax = Inf)
} else {
  NULL
}

# Reference-matrix position on the x_grid scale
x_ref <- alpha_ref * 10
y_ref <- approx(x_grid, f_env, xout = x_ref)$y

# ----------------------------
# Build plot

p <- ggplot(plot_df, aes(x = x, y = value,
                         colour    = Metric,
                         linetype  = Metric,
                         linewidth = Metric)) +
  
  # Confidence plateau shading
  { if (!is.null(conf_rect))
    geom_rect(data = conf_rect,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              inherit.aes = FALSE,
              fill = "#B2182B", alpha = 0.08) } +
  
  # Metric lines
  geom_line() +
  
  # Vertical reference lines
  geom_vline(xintercept = x_star, colour = "#B2182B",
             linetype = "dotted", linewidth = 0.7) +
  geom_vline(xintercept = x_ref,  colour = "grey30",
             linetype = "dotted", linewidth = 0.7) +
  
  # Optimal point (x*, A)
  annotate("point", x = x_star, y = A,
           shape = 19, size = 3, colour = "#B2182B") +
  annotate("text",  x = x_star, y = A,
           label = "x*", hjust = -0.4, vjust = 0.4,
           size = 3.2, colour = "#B2182B", fontface = "italic") +
  
  # Reference design-matrix point
  annotate("point", x = x_ref, y = y_ref,
           shape = 17, size = 3, colour = "grey20") +
  annotate("text",  x = x_ref, y = y_ref,
           label = "Ref.", hjust = -0.35, vjust = 0.4,
           size = 3.2, colour = "grey20") +
  
  # Scales
  scale_colour_manual(values = metric_colours,    name = NULL) +
  scale_linetype_manual(values = metric_linetypes, name = NULL) +
  scale_linewidth_manual(values = metric_linewidths, name = NULL) +
  
  # Labels
  labs(
    title    = "Hoegn Index — neurological data",
    subtitle = expression(italic("ncomp") ~ "= 1  |  shaded region: 99 % plateau"),
    x        = expression(italic(x) ~ "(α × 10)"),
    y        = "Metric value"
  ) +
  
  # Academic theme
  theme_classic(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle    = element_text(size = 10, colour = "grey40", hjust = 0,
                                    margin = margin(b = 8)),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 10, colour = "grey20"),
    axis.line        = element_line(colour = "grey30", linewidth = 0.4),
    axis.ticks       = element_line(colour = "grey30", linewidth = 0.4),
    legend.position  = c(0.98, 0.02),
    legend.justification = c("right", "bottom"),
    legend.background = element_rect(fill = "white", colour = "grey80",
                                     linewidth = 0.3),
    legend.key.width  = unit(1.6, "lines"),
    legend.text       = element_text(size = 9.5),
    legend.spacing.y  = unit(0.2, "lines"),
    panel.grid.major  = element_line(colour = "grey92", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    plot.margin       = margin(12, 16, 8, 8)
  ) +
  
  # Keep legend guides merged across colour/linetype/linewidth
  guides(
    colour    = guide_legend(override.aes = list(linewidth = c(.8,.8,.8,.8,1.3))),
    linetype  = guide_legend(),
    linewidth = "none"
  )

print(p)

# Optional: save to file
# ggsave("hoegn_index_plot.pdf", p, width = 7, height = 4.5, dpi = 300)
ggsave("hoegn_index_plot_neurologicalData_ncomp1.png", p, width = 7, height = 4.5, dpi = 500)