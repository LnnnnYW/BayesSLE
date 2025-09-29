options(repos = c(CRAN = "https://cran.rstudio.com"))  

pkgs <- c(
  "dplyr", "tidyr", "ggplot2", "MASS", "keras", "reticulate", "data.table",
  "doParallel", "foreach", "shapley", "glmnet", "lime", "patchwork",
  "future.apply", "future", "cmdstanr"
)

to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install)

invisible(lapply(pkgs, library, character.only = TRUE))