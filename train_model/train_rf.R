# train_rf.R
train_rf <- function(X, y) {
  library(ranger)
  df <- as.data.frame(X)
  df$.outcome <- y
  ranger(
    formula      = .outcome ~ .,
    data         = df,
    num.trees    = 500,
    respect.unordered.factors = "order"
  )
}

# make sure predict() returns a numeric vector
predict.ranger <- function(object, newdata, ...) {
  df_new    <- as.data.frame(newdata)
  preds_out <- predict(object, data = df_new)$predictions
  as.numeric(preds_out)
}
