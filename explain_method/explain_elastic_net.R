explain_elastic_net <- function(
  X, y, model, K = 30, seed = 123,
  mask_prob = 0.5, ..., active_idx = NULL
) {
  library(glmnet)

  if (is.null(active_idx)) active_idx <- attr(X, "active_idx")

  set.seed(seed) 

  p <- ncol(X)
  n_samples <- 2000L

  # pick a random x0
  x0_idx  <- sample.int(nrow(X), 1L)
  x0_vec  <- as.numeric(X[x0_idx, ])
  baseline <- colMeans(X)

  # build perturbations with mask_prob
  Z_binary    <- matrix(1L, n_samples, p)
  X_perturbed <- matrix(rep(x0_vec, times = n_samples),
                        nrow = n_samples, ncol = p, byrow = TRUE)

  for (j in seq_len(n_samples)) {
    # z=1 keep x0; z=0 set baseline
    mask_j <- rbinom(p, 1L, 1 - mask_prob)
    Z_binary[j, ] <- mask_j
    if (any(mask_j == 0L)) {
      X_perturbed[j, mask_j == 0L] <- baseline[mask_j == 0L]
    }
  }

  # get black-box predictions
  if (inherits(model, "xgb.Booster")) {
    y_pred <- as.numeric(predict(model, as.matrix(X_perturbed)))
  } else if (inherits(model, "cv.glmnet")) {
    y_pred <- as.numeric(predict(model, newx = as.matrix(X_perturbed), s = "lambda.min"))
  } else if (inherits(model, "keras.src.models.sequential.Sequential") ||
             inherits(model, "keras.src.models.functional.Functional")) {
    y_pred <- as.numeric(model %>% predict(as.matrix(X_perturbed)))
  } else {
    y_pred <- as.numeric(predict(model, X_perturbed))
  }

  # distance-based kernel weights (Euclidean)
  dist_vec <- sqrt(rowSums((X_perturbed - matrix(x0_vec, n_samples, p, byrow = TRUE))^2))
  kernel_width <- 0.75 * mean(dist_vec)
  if (!is.finite(kernel_width) || kernel_width <= 0) kernel_width <- 1.0  # guard
  weights <- exp(- (dist_vec^2) / (kernel_width^2))

  # protect against constant y_pred
  if (abs(stats::sd(y_pred, na.rm = TRUE)) < 1e-12) {
    y_pred <- y_pred + rnorm(length(y_pred), 0, 1e-6)
  }

  cv_fit <- glmnet::cv.glmnet(
    x = Z_binary, y = y_pred, weights = weights,
    alpha = 0.5, intercept = TRUE, standardize = TRUE, nfolds = 5
  )

  best_lambda <- cv_fit$lambda.min
  imp_scores  <- abs(as.numeric(coef(cv_fit, s = best_lambda))[-1])
  imp_scores[is.na(imp_scores)] <- 0

  K_use <- min(K, p)
  ord   <- order(imp_scores, decreasing = TRUE)
  topk_vars <- ord[seq_len(K_use)]

  if (!is.null(active_idx)) {
    rec <- sum(topk_vars %in% active_idx)
    message(sprintf("[elastic_net] x0=%d | mask_prob=%.2f | recall=%d/%d",
                    x0_idx, mask_prob, rec, length(active_idx)))
  }

  list(topk_vars = topk_vars, importance = imp_scores, x0_index = x0_idx)
}
