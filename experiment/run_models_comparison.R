# run_compare_model.R

source("generate_sparse_data.R")
source("evaluation/eval_recall.R")
source("evaluation/output_table.R")
source("evaluation/plot_summary.R")

source("experiment/run_experiment_grid.R")  

source("train_model/train_dnn.R")
source("train_model/train_lasso.R")
source("train_model/train_spinn.R")
source("train_model/train_lin_interact.R")
source("train_model/train_rf.R")
source("train_model/train_svm.R")

source("explain_method/explain_bayes_sle.R")
source("explain_method/explain_elastic_net.R")
source("explain_method/explain_lime_dnn.R")
source("explain_method/explain_shap_dnn.R")
source("explain_method/explain_wrapper.R")


suppressPackageStartupMessages({
  library(future)
  library(future.apply)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grid) 
})

run_compare_sle <- function(
  model_types  = c("lin_interact","lasso","dnn","spinn","rf","svm"),
  p            = 100,
  n            = 200,
  s            = 30,
  H2           = 0.5,
  K            = 30,
  seed0        = 100,
  n_repeats    = 100,
  out_dir_base = "results/sle_compare",
  scenario     = "nonlinear"
) {
  # parallel
  future::plan(multisession, workers = min(n_repeats, availableCores()))
  on.exit(future::plan(sequential), add = TRUE)

  # experiments
  results_list <- lapply(model_types, function(mt) {
    message("Running Bayes-SLE on model: ", mt)
    df_mt <- future.apply::future_lapply(seq_len(n_repeats), function(i) {
      res <- run_experiment_grid(
        p            = p,
        n            = n,
        s            = s,
        H2           = H2,
        model_types  = mt,
        interpreters = "bayes_sle",
        K            = K,
        seed         = seed0 + i,
        repeat_id    = i,
        scenario     = scenario
      )
      tbl <- res$table
      tbl$model_type <- mt
      tbl
    }, future.seed = TRUE)
    bind_rows(df_mt)
  })
  all_df <- bind_rows(results_list)

  # save raw
  dir.create(out_dir_base, recursive = TRUE, showWarnings = FALSE)
  write.csv(all_df,
            file = file.path(out_dir_base, "sle_compare_results.csv"),
            row.names = FALSE)

  stopifnot(all(c("model_type","recall_at_k","precision_at_k","fp_at_k","f1") %in% names(all_df)))
  gold_mean <- data.frame(
    metric = c("recall_at_k","precision_at_k","fp_at_k","f1"),
    mean   = c(mean(all_df$recall_at_k,    na.rm=TRUE),
               mean(all_df$precision_at_k, na.rm=TRUE),
               mean(all_df$fp_at_k,        na.rm=TRUE),
               mean(all_df$f1,             na.rm=TRUE))
  )

  long_df <- all_df %>%
    dplyr::select(model_type, recall_at_k, precision_at_k, fp_at_k, f1) %>%
    tidyr::pivot_longer(
      cols      = c(recall_at_k, precision_at_k, fp_at_k, f1),
      names_to  = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric     = factor(metric, levels = c("recall_at_k","precision_at_k","fp_at_k","f1")),
      model_type = factor(model_type, levels = model_types),
      value      = as.numeric(value)
    )

  check_mean <- long_df %>%
    group_by(metric) %>% summarise(mean = mean(value, na.rm=TRUE), .groups="drop")
  stopifnot(all(abs(check_mean$mean - gold_mean$mean[match(check_mean$metric, gold_mean$metric)]) < 1e-12))

  accent        <- "#D55E00"   # red for mean/CI
  violin_fill   <- "#F3E1D6"   # soft neutral
  violin_alpha  <- 0.35
  box_alpha     <- 0.22

  # plot
  p <- ggplot(long_df, aes(x = model_type, y = value)) +
    geom_violin(fill = violin_fill, alpha = violin_alpha, color = "black",
                linewidth = 0.5, trim = FALSE) +
    geom_boxplot(width = 0.16, outlier.shape = NA,
                 color = "black", fill = "white", alpha = box_alpha, linewidth = 0.55,
                 median.linewidth = 0.9, median.colour = "black") +
    # 95% CI (red)
    stat_summary(fun.data = function(x){
        m <- mean(x); se <- sd(x)/sqrt(length(x))
        data.frame(y = m, ymin = m - 1.96*se, ymax = m + 1.96*se)
      },
      geom = "errorbar", width = 0.16, color = accent, linewidth = 0.75
    ) +
    stat_summary(fun = mean,
                 geom = "point",
                 shape = 23, size = 2.2, fill = "white", color = accent, stroke = 0.9) +
    facet_wrap(~ metric, scales = "free_y", ncol = 2,
               labeller = as_labeller(c(
                 recall_at_k    = "Recall@K",
                 precision_at_k = "Precision@K",
                 fp_at_k        = "FP@K",
                 f1             = "F1"
               ))) +
    labs(title = "Bayes-SLE Performance Across Black-Box Models",
         x = "Black-Box Model", y = NULL) +
    theme_classic(base_size = 16) +
    theme(
      plot.title         = element_text(face = "bold", hjust = 0.5),
      strip.text         = element_text(face = "bold", size = 14),
      axis.text.x        = element_text(angle = 25, vjust = 1, hjust = 1, size = 12),
      axis.text.y        = element_text(size = 12),
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.4),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.spacing      = unit(1.1, "lines"),
      legend.position    = "none"
    )

  ggsave(file.path(out_dir_base, "sle_compare_plot.pdf"), p, width = 10, height = 8)
  invisible(all_df)
}

run_compare_sle(
  model_types  = c("lin_interact","lasso","dnn","spinn","rf","svm"),
  p            = 100,
  n            = 200,
  s            = 30,
  H2           = 0.5,
  K            = 30,
  seed0        = 100,
  n_repeats    = 500,
  out_dir_base = "sle_compare_results",
  scenario     = "nonlinear"
)
