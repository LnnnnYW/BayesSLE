# train_svm.R
train_svm <- function(X, y) {
  library(e1071)
  svm(
    x      = X,
    y      = y,
    type   = "eps-regression",
    kernel = "radial",
    scale  = TRUE
  )
}

# default predict.svm returns numeric for regression
