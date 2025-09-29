# evaluate function - recall/precision/false positive
eval_recall <- function(selected, true) {
  if (length(selected) == 0 || all(is.na(selected))) return(NA_real_)
  sum(selected %in% true) / length(true)            
}

eval_precision <- function(selected, true) {
  if (length(selected) == 0 || all(is.na(selected))) return(NA_real_)
  sum(selected %in% true) / length(selected)
}

eval_fp <- function(selected, true) {
  if (length(selected) == 0 || all(is.na(selected))) return(NA_real_)
  sum(!(selected %in% true))
}
