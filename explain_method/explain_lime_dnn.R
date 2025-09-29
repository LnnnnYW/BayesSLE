explain_lime_dnn <- function(
  X, y, model, K = 30, seed = 123,
  sparsity = 2, n_x0 = 20, n_samples = 2000,
  ..., active_idx = NULL
) {
  if (is.null(active_idx)) active_idx <- attr(X, "active_idx")
  set.seed(seed)

  n <- nrow(X); p <- ncol(X)
  sparsity <- min(sparsity, p)
  selected_mat <- matrix(0L, nrow = p, ncol = n_x0)
  predict_fun <- make_predict_fun(model)

  for (i in seq_len(n_x0)) {
    x0_idx  <- sample.int(n, 1L)
    x0_vec  <- as.numeric(X[x0_idx, ])
    baseline <- colMeans(X)

    Z_binary    <- matrix(1L, n_samples, p)
    X_perturbed <- matrix(rep(x0_vec, times = n_samples),
                          nrow = n_samples, ncol = p, byrow = TRUE)

    for (j in seq_len(n_samples)) {
      mask_idx <- sample.int(p, sparsity)
      Z_binary[j, mask_idx] <- 0L
      X_perturbed[j, mask_idx] <- baseline[mask_idx]
    }

    y_pred <- predict_fun(X_perturbed)

    dist_vec <- sqrt(rowSums((X_perturbed - matrix(x0_vec, n_samples, p, byrow = TRUE))^2))
    kernel_width <- 0.75 * mean(dist_vec)
    if (!is.finite(kernel_width) || kernel_width <= 0) kernel_width <- 1.0
    weights <- exp(- (dist_vec^2) / (kernel_width^2))

    if (abs(stats::sd(y_pred, na.rm = TRUE)) < 1e-12) {
      y_pred <- y_pred + rnorm(length(y_pred), 0, 1e-6)
    }

    fit <- glmnet::cv.glmnet(
      x = Z_binary, y = y_pred, weights = weights,
      alpha = 1, intercept = TRUE, standardize = TRUE, nfolds = 5
    )
    imp_scores <- abs(as.numeric(glmnet::coef.glmnet(fit, s = "lambda.min")[-1]))
    imp_scores[is.na(imp_scores)] <- 0

    K_use <- min(K, p)
    selected_idx <- order(imp_scores, decreasing = TRUE)[seq_len(K_use)]

    if (length(selected_idx) > 0L) selected_mat[selected_idx, i] <- 1L

    if (!is.null(active_idx) && length(active_idx) > 0) {
      rec_i <- sum(selected_idx %in% active_idx)
      message(sprintf("[lime] x0 %2d: recall = %d/%d", i, rec_i, length(active_idx)))
    }
  }

  local_freq <- rowMeans(selected_mat)
  ord_local  <- order(local_freq, decreasing = TRUE)
  topk_local <- head(ord_local, min(K, p))

  list(
    selected     = topk_local,
    local_freq   = local_freq,
    topk_local   = topk_local,
    selected_mat = selected_mat
  )
}
