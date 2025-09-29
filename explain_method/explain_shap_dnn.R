explain_shap_dnn <- function(
  model, X, active_idx = NULL,
  n_x0 = 20, n_samples = 2000, K = 30,
  use_inout = FALSE, in_prob = 0.5, out_prob = 0.5, S = NULL,
  method = c("lasso", "ridge"), seed = 123
) {
  if (is.null(active_idx)) active_idx <- attr(X, "active_idx")
  set.seed(seed)

  n <- nrow(X); p <- ncol(X)
  method <- match.arg(method)
  recalls      <- integer(n_x0)
  selected_mat <- matrix(0L, nrow = p, ncol = n_x0)
  predict_fun  <- make_predict_fun(model)

  for (i in seq_len(n_x0)) {
    x0_idx  <- sample.int(n, 1L)
    x0_vec  <- as.numeric(X[x0_idx, ])
    baseline <- colMeans(X)

    Z_binary    <- matrix(1L, n_samples, p)
    X_perturbed <- matrix(rep(x0_vec, times = n_samples),
                          nrow = n_samples, ncol = p, byrow = TRUE)

    for (j in seq_len(n_samples)) {
      if (use_inout && !is.null(S)) {
        prob <- rep(out_prob, p); prob[S] <- in_prob
        z <- rbinom(p, 1L, prob)
      } else {
        z <- rbinom(p, 1L, 0.5)
      }
      Z_binary[j, ] <- z
      if (any(z == 0L)) X_perturbed[j, z == 0L] <- baseline[z == 0L]
    }

    y_pred <- predict_fun(X_perturbed)
    if (p < 2L) next

    if (abs(stats::sd(y_pred, na.rm = TRUE)) < 1e-12) {
      y_pred <- y_pred + rnorm(length(y_pred), 0, 1e-6)
    }

    if (method == "lasso") {
      cv_fit <- glmnet::cv.glmnet(
        x = Z_binary, y = y_pred,
        alpha = 1, standardize = TRUE, intercept = TRUE
      )
    } else {
      cv_fit <- glmnet::cv.glmnet(
        x = Z_binary, y = y_pred,
        alpha = 0, standardize = TRUE, intercept = TRUE
      )
    }

    imp_scores <- abs(as.numeric(glmnet::coef.glmnet(cv_fit, s = "lambda.min")[-1]))
    imp_scores[is.na(imp_scores)] <- 0

    K_use <- min(K, p)
    selected_idx <- order(imp_scores, decreasing = TRUE)[seq_len(K_use)]
    if (length(selected_idx) > 0L) selected_mat[selected_idx, i] <- 1L

    if (!is.null(active_idx) && length(active_idx) > 0L) {
      recalls[i] <- sum(selected_idx %in% active_idx)
      message(sprintf("[shap] x0 %2d: recall = %d/%d",
                      i, recalls[i], length(active_idx)))
    }
  }

  local_freq <- rowMeans(selected_mat)
  ord_local  <- order(local_freq, decreasing = TRUE)
  topk_local <- head(ord_local, min(K, p))

  final_recall <- if (!is.null(active_idx) && length(active_idx) > 0L) {
    sum(topk_local %in% active_idx)
  } else NA_integer_

  list(
    selected     = topk_local,
    local_freq   = local_freq,
    recalls      = recalls,
    mean_recall  = if (length(active_idx) > 0L) mean(recalls, na.rm = TRUE) else NA_real_,
    final_recall = final_recall,
    selected_mat = selected_mat
  )
}
