# train_lasso.R
# use glmnet linear LASSO to construct a black-box 

train_lasso <- function(X, y, seed = 42) {
  library(glmnet); set.seed(seed)
  cv <- cv.glmnet(X, y, alpha = 1, nfolds = 5, family = "gaussian")
  best_lambda <- cv$lambda.min
  fit <- glmnet(X, y, alpha = 1, lambda = best_lambda, family = "gaussian")
  attr(fit, "best_lambda") <- best_lambda
  class(fit) <- c("lasso_glmnet", class(fit))
  fit
}
predict.lasso_glmnet <- function(object, newx, ...)
  as.numeric(glmnet::predict.glmnet(object, newx, s = attr(object, "best_lambda")))
