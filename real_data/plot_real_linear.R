suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(forcats)
})

make_real_linear_plots <- function(
  out_dir,
  formats = c("pdf", "png"),
  weight_min_raw  = 0.10,
  weight_min_norm = NULL,
  topN_per_method = NULL,
  font_family     = NULL,
  use_showtext    = FALSE,
  width           = 8,
  height          = 5,
  dpi             = 300,
  title_infidelity = "Infidelity by method (linear model)",
  title_topvars    = "Top variables by method (linear model)",
  ylab_weight      = "mean |weight|",
  ylab_infidelity  = "infidelity (lower is better)"
) {
  sum_path <- file.path(out_dir, "summary_linear.csv")
  sel_path <- file.path(out_dir, "selected_linear.csv")
  if (!file.exists(sum_path)) stop("Cannot find: ", sum_path)
  if (!file.exists(sel_path)) stop("Cannot find: ", sel_path)

  df_sum <- readr::read_csv(sum_path, show_col_types = FALSE) %>%
    dplyr::filter(.data$model == "linear")

  df_sel <- readr::read_csv(sel_path, show_col_types = FALSE) %>%
    dplyr::filter(.data$model == "linear")

  keep_methods <- c("BayesSLE", "KernelSHAP", "LIME", "IG")
  df_sel <- df_sel %>%
    dplyr::filter(.data$method %in% keep_methods) %>%
    dplyr::filter(is.finite(.data$weight))

  # aggregate mean |weight| per method/var
  safe_max <- function(x) {
    m <- suppressWarnings(max(x, na.rm = TRUE))
    if (!is.finite(m) || m <= 0) 1 else m
  }

  agg <- df_sel %>%
    dplyr::mutate(w_abs = abs(.data$weight)) %>%
    dplyr::group_by(.data$method, .data$var) %>%
    dplyr::summarise(w_mean = mean(.data$w_abs, na.rm = TRUE), .groups = "drop") %>%
    dplyr::group_by(.data$method) %>%
    dplyr::mutate(w_norm = .data$w_mean / safe_max(.data$w_mean)) %>%
    dplyr::ungroup()

  filt <- agg
  if (!is.null(weight_min_raw))  filt <- filt %>% dplyr::filter(.data$w_mean >= weight_min_raw)
  if (!is.null(weight_min_norm)) filt <- filt %>% dplyr::filter(.data$w_norm >= weight_min_norm)
  if (!is.null(topN_per_method)) {
    filt <- filt %>%
      dplyr::group_by(.data$method) %>%
      dplyr::slice_max(order_by = .data$w_mean, n = topN_per_method, with_ties = FALSE) %>%
      dplyr::ungroup()
  }
  if (nrow(filt) == 0) {
    warning("No variables passed the weight filter. Try lowering 'weight_min_raw' or 'weight_min_norm'.")
  }

  filt <- filt %>%
    dplyr::group_by(.data$method) %>%
    dplyr::arrange(dplyr::desc(.data$w_mean), .by_group = TRUE) %>%
    dplyr::mutate(var = forcats::fct_reorder(.data$var, .data$w_mean)) %>%
    dplyr::ungroup()

  # plots
  p_inf <- ggplot(df_sum, aes(x = .data$method, y = .data$infidelity_mean)) +
    geom_col(width = 0.65) +
    geom_errorbar(aes(ymin = .data$infidelity_mean - .data$infidelity_sd,
                      ymax = .data$infidelity_mean + .data$infidelity_sd),
                  width = 0.2) +
    labs(x = NULL, y = ylab_infidelity, title = title_infidelity) +
    theme_bw(base_family = font_family) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(hjust = 0, face = "bold"))

  p_vars <- ggplot(filt, aes(x = .data$var, y = .data$w_mean)) +
    geom_col(width = 0.65) +
    coord_flip() +
    facet_wrap(~ .data$method, scales = "free_y") +
    labs(x = NULL, y = ylab_weight, title = title_topvars) +
    theme_bw(base_family = font_family) +
    theme(strip.background = element_blank(),
          plot.title = element_text(hjust = 0, face = "bold"))

  # helpers
  figs_dir <- file.path(out_dir, "figs")
  dir.create(figs_dir, showWarnings = FALSE, recursive = TRUE)

  if (use_showtext && requireNamespace("showtext", quietly = TRUE)) {
    showtext::showtext_auto()
  }

  has_cairo <- FALSE
  try(has_cairo <- isTRUE(capabilities("cairo")), silent = TRUE)

  save_plot <- function(p, name) {
    for (fmt in formats) {
      f <- file.path(figs_dir, paste0(name, ".", fmt))
      if (fmt == "pdf") {
        if (has_cairo) {
          ggplot2::ggsave(filename = f, plot = p,
                          width = width, height = height,
                          device = grDevices::cairo_pdf)
        } else {
          ggplot2::ggsave(filename = f, plot = p,
                          width = width, height = height,
                          device = grDevices::pdf)
        }
      } else {
        ggplot2::ggsave(filename = f, plot = p,
                        width = width, height = height, dpi = dpi)
      }
    }
  }

  save_plot(p_inf,  "linear_infidelity")
  save_plot(p_vars, "linear_topvars")

  invisible(list(
    p_inf   = p_inf,
    p_vars  = p_vars,
    data    = list(summary = df_sum, topvars = filt),
    figsdir = figs_dir
  ))
}
