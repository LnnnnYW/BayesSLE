# output_table.R

output_table <- function(res_df) {
  library(dplyr)
  
  summary_tbl <- res_df %>%
    group_by(p, s, H2, model, interpreter) %>%
    summarise(
      mean_recall = mean(recall_at_k, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(
      names_from = interpreter,
      values_from = mean_recall
    )
  
  return(summary_tbl)
}