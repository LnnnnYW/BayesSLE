source("generate_sparse_data.R")
source("evaluation/eval_recall.R")
source("evaluation/output_table.R")
source("evaluation/plot_summary.R")

source("experiment/run_experiment_grid.R")  
source("train_model/train_dnn.R")
source("train_model/train_lasso.R")
source("train_model/train_spinn.R")
source("train_model/train_lin_interact.R")

source("explain_method/explain_bayes_sle.R")
source("explain_method/explain_elastic_net.R")
source("explain_method/explain_lime.R")
source("explain_method/explain_lime_dnn.R")
source("explain_method/explain_shap.R")
source("explain_method/explain_shap_dnn.R")
source("explain_method/explain_wrapper.R")

run_interpreter_compare_high <- function(
  n_repeats    = 500,
  out_dir_base = "results/intp_sim_fmri",
  interpreters = c("bayes_sle","elastic_net","lime","shap"),
  model_type   = "lin_interact",
  p            = 100,
  n            = 200,
  s            = 30,
  H2           = 0.5,
  K            = 30,
  seed0        = 100,
  scenario     = "linear" 
) {
  # 0. setup
  suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(patchwork)
    library(data.table)
    library(future)
    library(future.apply)
  })

  if (scenario == "all") {
    all_scenarios <- c(
      "linear",
      "high_snr",
      "low_snr",
      "sparse",
      "correlated",
      "corr_int",
      "nonlinear",
      "heteroscedastic",
      "highdim_smalln"
    )
    return(
      invisible(
        lapply(all_scenarios, function(sc) {
          run_interpreter_compare_high(
            n_repeats    = n_repeats,
            out_dir_base = file.path(out_dir_base, sc),
            interpreters = interpreters,
            model_type   = model_type,
            p            = p,
            n            = n,
            s            = s,
            H2           = H2,
            K            = K,
            seed0        = seed0,
            scenario     = sc
          )
        })
      )
    )
  }

  future::plan(multisession, workers = min(n_repeats, future::availableCores()))

  # 1. run experiments in parallel
  res_list <- future.apply::future_lapply(seq_len(n_repeats), function(i) {
    run_experiment_grid(
      p            = p,
      n            = n,
      s            = s,
      H2           = H2,
      model_types  = model_type,
      interpreters = interpreters,
      K            = K,
      seed         = seed0 + i,
      repeat_id    = i,
      scenario     = scenario
    )
  }, future.seed = TRUE)

  # 2. combine and sanity check
  all_df <- bind_rows(lapply(res_list, `[[`, "table"))
  print(all_df %>% count(interpreter))
  stopifnot(all(all_df %>% count(interpreter) %>% pull(n) == n_repeats))

  # 3. save raw results
  dir.create(out_dir_base, recursive = TRUE, showWarnings = FALSE)
  data.table::fwrite(
    all_df,
    file.path(out_dir_base, "all_sim_results.csv")
  )

  # 4. reshape to long format
  long_df <- all_df %>%
    pivot_longer(
      cols      = c(recall_at_k, precision_at_k, fp_at_k, f1),
      names_to  = "metric",
      values_to = "value"
    )

  # 5. plot function
  plot_metric <- function(metric_name, ylab) {
    df <- filter(long_df, metric == metric_name)
    ggplot(df, aes(x = interpreter, y = value, fill = interpreter)) +
      geom_violin(trim = FALSE, width = 0.8, alpha = 0.3, color = NA) +
      geom_jitter(width = 0.15, size = 0.6, alpha = 0.1, color = "grey20") +
      geom_boxplot(width = 0.2, outlier.shape = NA,
                   color = "black", linewidth = 0.6, alpha = 0.6) +
      stat_summary(fun.data = mean_se,
                   geom      = "errorbar",
                   width     = 0.1,
                   color     = "black",
                   linewidth = 0.8) +    
      stat_summary(fun       = mean,
                   geom      = "point",
                   shape     = 23,
                   size      = 3,
                   fill      = "white",
                   color     = "black") +
      stat_summary(
        fun.data  = function(x) {
          m  <- mean(x)
          se <- sd(x) / sqrt(length(x))
          data.frame(y = m, ymin = m - 1.96*se, ymax = m + 1.96*se)
        },
        geom       = "errorbar",
        width      = 0.2,
        color      = "red",
        linewidth  = 0.5       
      ) +
      labs(title = ylab, x = NULL, y = ylab) +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "none",
        plot.title      = element_text(face = "bold", hjust = 0.5),
        axis.text.x     = element_text(angle = 25, vjust = 1, hjust = 1)
      )
  }

  # 6. draw panels
  p1 <- plot_metric("recall_at_k",    "Recall@K")
  p2 <- plot_metric("precision_at_k", "Precision@K")
  p3 <- plot_metric("fp_at_k",        "FP@K")
  p4 <- plot_metric("f1",             "F1")
  p_all <- (p1 | p2) / (p3 | p4)

  # 7. save figure
  ggsave(
    filename = file.path(out_dir_base, "compare_all_metrics.pdf"),
    plot     = p_all,
    width    = 12,
    height   = 10
  )

  # 8. variable selection heatmap
  set.seed(seed0 + 1)
  active_idx2 <- sample.int(p, s)
  X2 <- matrix(rnorm(n * p), nrow = n, ncol = p)
  attr(X2, "active_idx") <- active_idx2
  beta2 <- numeric(p); beta2[active_idx2] <- rnorm(s)
  var_sig2 <- var(as.vector(X2 %*% beta2))
  sigma22 <- var_sig2 * (1 - H2) / H2
  y2 <- as.numeric(X2 %*% beta2 + rnorm(n, sd = sqrt(sigma22)))

  script1 <- list.files(".", "train_lin_interact\\.R$", recursive=TRUE, full.names=TRUE)[1]
  source(script1)
  model2 <- train_lin_interact(X2, y2)

  # run each interpreter once
  sel_list <- lapply(interpreters, function(m) {
    out2 <- explain_wrapper(
      X          = X2,
      y          = y2,
      model      = model2,
      method     = m,
      K          = K,
      seed       = seed0 + 1,
      active_idx = active_idx2
    )
    paste0("V", out2$selected)
  })
  names(sel_list) <- interpreters

  # build comparison table
  vars <- paste0("V", seq_len(p))
  df_cmp <- data.frame(
    variable = vars,
    active   = vars %in% paste0("V", active_idx2),
    stringsAsFactors = FALSE
  )
  for(m in interpreters) {
    df_cmp[[m]] <- vars %in% sel_list[[m]]
  }

  # pivot and plot improved heatmap
  df_long2 <- df_cmp %>%
    arrange(desc(active)) %>%
    pivot_longer(
      cols      = all_of(interpreters),
      names_to  = "method",
      values_to = "selected"
    ) %>%
    mutate(
      status = case_when(
        active & selected   ~ "TP",
        !active & selected  ~ "FP",
        TRUE                ~ "None"
      ),
      status = factor(status, levels = c("TP","FP","None"))
    )

  p_tile2 <- ggplot(df_long2, aes(x = method, y = variable, fill = status)) +
    geom_tile(color = "white") +
    scale_fill_manual(
      name   = "Selection status",
      values = c(TP="forestgreen", FP="firebrick", None="white"),
      labels = c(TP="True Positive", FP="False Positive", None="Not selected")
    ) +
    facet_grid(
      active ~ ., scales = "free_y", space = "free_y",
      switch = "y",
      labeller = labeller(active = c(`TRUE`="Active", `FALSE`="Inactive"))
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.y     = element_blank(),
      axis.ticks.y    = element_blank(),
      panel.grid      = element_blank(),
      strip.placement = "outside",
      strip.text.y    = element_text(angle = 0, face = "bold")
    ) +
    labs(
      title = "Variable Selection: True vs False Positives",
      x     = "Interpreter",
      y     = NULL
    )

  ggsave(
    file.path(out_dir_base, "variable_selection_heatmap.pdf"),
    p_tile2, width = 6, height = 8
  )

  invisible(all_df)
}

# example: run all 9 regression scenarios
run_interpreter_compare_high(
  n_repeats    = 100,
  out_dir_base = "interpreter_compare_results",
  scenario     = "all"
)
