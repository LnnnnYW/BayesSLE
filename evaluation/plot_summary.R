# plot_summary.R

plot_summary <- function(res_df) {
  library(ggplot2)
  
  ggplot(res_df, aes(x = interpreter, y = recall_at_k, fill = model)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
    facet_grid(H2 ~ p, labeller = label_both) +
    labs(
      title = "Recall@K across Models and Interpreters",
      x = "Interpreter",
      y = "Recall@K"
    ) +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

