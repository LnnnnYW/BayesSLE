# code/models/train_lin_interact.R
train_lin_interact <- function(X, y, seed = 42,
                               alpha = 1,        # 1=LASSO, 0=ridge
                               nfolds = 5) {

  suppressPackageStartupMessages(library(glmnet))
  set.seed(seed)

  X_df <- as.data.frame(X)
  colnames(X_df) <- paste0("V", seq_len(ncol(X_df)))

  mm <- model.matrix(~ .^2, data = X_df)[, -1, drop = FALSE]

  cv <- cv.glmnet(mm, y, alpha = alpha,
                  nfolds = nfolds, family = "gaussian")
  best_lambda <- cv$lambda.min

  fit <- glmnet(mm, y, alpha = alpha,
                lambda = best_lambda, family = "gaussian")

  res <- list(fit = fit,
              cols = colnames(mm)) 
  class(res) <- "lin_interaction"
  attr(res, "best_lambda") <- best_lambda
  res
}


predict.lin_interaction <- function(object, newx, ...) {
  X_df <- as.data.frame(newx)
  colnames(X_df) <- paste0("V", seq_len(ncol(X_df)))
  mm <- model.matrix(~ .^2, data = X_df)[, -1, drop = FALSE]

  missing <- setdiff(object$cols, colnames(mm))
  if (length(missing))
    mm <- cbind(mm,
                matrix(0, nrow(mm), length(missing),
                       dimnames = list(NULL, missing)))
  mm <- mm[, object$cols, drop = FALSE]

  as.numeric(glmnet::predict.glmnet(object$fit, mm,
                                    s = attr(object, "best_lambda")))
}