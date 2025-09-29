explain_wrapper <- function(
  X, y, model,
  method = c("bayes_sle", "elastic_net", "lime", "shap"),
  K = 30,
  seed = 123,
  spinn_model = NULL,
  active_idx = NULL,
  ...
) {
  method <- match.arg(method)

  if (!is.null(active_idx)) {
    attr(X, "active_idx") <- active_idx
  }

  if (method == "bayes_sle") {
    args <- list(X = X, y = y, model = model,
                 K = K, seed = seed, spinn_model = spinn_model)
    if ("active_idx" %in% names(formals(explain_bayes_sle))) {
      args$active_idx <- active_idx
    }
    res <- do.call(explain_bayes_sle, args)
    return(list(
      type           = "bayes_sle",
      selected       = res$topk_vars,
      posterior      = res$posterior,
      posterior_mean = res$posterior$mean,
      posterior_sd   = res$posterior$sd,
      best_lambda    = res$best_lambda,
      spinn_score    = res$spinn_score,
      x0_index       = res$x0_index
    ))
  }

  if (method == "elastic_net") {
  res <- explain_elastic_net(
    X, y, model,
    K         = K,
    seed      = seed,
    mask_prob = 0.5,   
    active_idx = active_idx,
    ...
  )
  return(list(
    type       = "elastic_net",
    selected   = res$topk_vars,
    importance = res$importance,
    x0_index   = res$x0_index
  ))
}

  if (method == "lime") {
    res <- explain_lime_dnn(X, y, model, K = K, seed = seed,
                            active_idx = active_idx, ...)
    return(list(
      type       = "lime",
      selected   = res$topk_local,
      importance = res$local_freq
    ))
  }

  if (method == "shap") {
  res <- explain_shap_dnn(
    model     = model,
    X         = X,
    active_idx = active_idx,
    K         = K,
    seed      = seed,
    ...
  )
  return(list(
    type       = "shap",
    selected   = res$selected,      
    importance = res$local_freq
  ))
}

  stop(sprintf("Unknown interpreter: %s", method))
}
make_predict_fun <- function(model) {
  force(model)
  function(newdata) {
    as.numeric(predict(model, newx = as.matrix(newdata)))
  }
}

predict.lasso_glmnet <- function(object, newx, ...)
  glmnet::predict.glmnet(object, newx,
                         s = attr(object, "best_lambda") %||% "lambda.min")
predict.glmnet <- function(object, newx, ...)
  glmnet::predict.glmnet(object, newx,
                         s = attr(object, "best_lambda") %||% "lambda.min")