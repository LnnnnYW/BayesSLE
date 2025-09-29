# run_experiment_grid.R
# Batch run simulations across combinations of models 

source("explain_method/explain_wrapper.R")
source("train_model/train_dnn.R")
source("train_model/train_lasso.R")
source("train_model/train_spinn.R")
source("train_model/train_xgboost.R")
source("train_model/train_lin_interact.R")
source("train_model/train_rf.R")
source("train_model/train_svm.R")

# run_experiment_grid()
run_experiment_grid <- function(
  p, n, s, H2,
  model_types,   
  interpreters,
  K, seed, repeat_id,
  scenario = c(
    "linear", "high_snr", "low_snr", "sparse", "correlated",
    "corr_int", "nonlinear", "heteroscedastic", "highdim_smalln"
  )
) {
  # match the chosen scenario
  scenario <- match.arg(scenario)

  # 1. setup
  suppressPackageStartupMessages({ library(glmnet); library(dplyr) })
  set.seed(seed)

  # 2. simulate data according to scenario
  if (scenario == "linear") {
    active_idx <- sample.int(p, s)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    attr(X, "active_idx") <- active_idx
    beta_true  <- numeric(p); beta_true[active_idx] <- rnorm(s)
    var_signal <- var(as.vector(X %*% beta_true))
    sigma2     <- var_signal * (1 - H2) / H2
    y <- as.numeric(X %*% beta_true + rnorm(n, sd = sqrt(sigma2)))
  }
  else if (scenario == "high_snr") {
    active_idx <- sample.int(p, s)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    attr(X, "active_idx") <- active_idx
    beta_true  <- numeric(p); beta_true[active_idx] <- rnorm(s)
    var_signal <- var(as.vector(X %*% beta_true))
    sigma2     <- var_signal * (1 - 0.8) / 0.8
    y <- as.numeric(X %*% beta_true + rnorm(n, sd = sqrt(sigma2)))
  }
  else if (scenario == "low_snr") {
    active_idx <- sample.int(p, s)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    attr(X, "active_idx") <- active_idx
    beta_true  <- numeric(p); beta_true[active_idx] <- rnorm(s)
    var_signal <- var(as.vector(X %*% beta_true))
    sigma2     <- var_signal * (1 - 0.2) / 0.2
    y <- as.numeric(X %*% beta_true + rnorm(n, sd = sqrt(sigma2)))
  }
  else if (scenario == "sparse") {
    active_idx <- sample.int(p, s)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    attr(X, "active_idx") <- active_idx
    beta_true  <- numeric(p); beta_true[active_idx] <- rnorm(s)
    var_signal <- var(as.vector(X %*% beta_true))
    sigma2     <- var_signal * (1 - H2) / H2
    y <- as.numeric(X %*% beta_true + rnorm(n, sd = sqrt(sigma2)))
  }
  else if (scenario == "correlated") {
    active_idx <- sample.int(p, s)
    rho <- 0.8
    Sigma <- toeplitz(c(1, rep(rho, p - 1)))
    L <- chol(Sigma)
    Z <- matrix(rnorm(n * p), n, p)
    X <- Z %*% L
    attr(X, "active_idx") <- active_idx
    beta_true  <- numeric(p); beta_true[active_idx] <- rnorm(s)
    var_signal <- var(as.vector(X %*% beta_true))
    sigma2     <- var_signal * (1 - H2) / H2
    y <- as.numeric(X %*% beta_true + rnorm(n, sd = sqrt(sigma2)))
  }
  else if (scenario == "corr_int") {
    active_idx <- sample.int(p, s)
    rho <- 0.8
    Sigma <- toeplitz(c(1, rep(rho, p - 1)))
    L <- chol(Sigma)
    Z <- matrix(rnorm(n * p), n, p)
    X <- Z %*% L
    attr(X, "active_idx") <- active_idx
    beta_true <- numeric(p); beta_true[active_idx] <- rnorm(s)
    half <- floor(s/2)
    idx1  <- active_idx[1:half]
    idx2  <- active_idx[(half+1):(2*half)]
    gamma <- rnorm(half)
    sig_main <- X %*% beta_true
    sig_int  <- rowSums(
      X[, idx1, drop=FALSE] *
      X[, idx2, drop=FALSE] *
      matrix(gamma, nrow=n, ncol=half, byrow=TRUE)
    )
    var_tot <- var(as.vector(sig_main + sig_int))
    sigma2  <- var_tot * (1 - H2) / H2
    y <- as.numeric(0.6*sig_main + 0.4*sig_int + rnorm(n, sd=sqrt(sigma2)))
  }
  else if (scenario == "nonlinear") {
    active_idx <- sample.int(p, s)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    attr(X, "active_idx") <- active_idx
    beta_true <- numeric(p); beta_true[active_idx] <- rnorm(s)
    beta2 <- rnorm(s)
    lin_term <- X %*% beta_true
    sq_term  <- rowSums((X[, active_idx]^2) * matrix(beta2, nrow=n, ncol=s, byrow=TRUE))
    var_tot  <- var(as.vector(lin_term + sq_term))
    sigma2   <- var_tot * (1 - H2) / H2
    y <- lin_term + sq_term + rnorm(n, sd=sqrt(sigma2))
  }
  else if (scenario == "heteroscedastic") {
    active_idx <- sample.int(p, s)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    attr(X, "active_idx") <- active_idx
    beta_true <- numeric(p); beta_true[active_idx] <- rnorm(s)
    alpha <- runif(p, -0.5, 0.5)
    mu <- as.vector(X %*% beta_true)
    sigma_i <- exp(X %*% alpha)
    y <- mu + rnorm(n, sd=sigma_i)
  }
  else if (scenario == "highdim_smalln") {
    active_idx <- sample.int(p, s)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    attr(X, "active_idx") <- active_idx
    beta_true  <- numeric(p); beta_true[active_idx] <- rnorm(s)
    var_signal <- var(as.vector(X %*% beta_true))
    sigma2     <- var_signal * (1 - H2) / H2
    y <- as.numeric(X %*% beta_true + rnorm(n, sd = sqrt(sigma2)))
  }

  # 3. decide the script 
  script_name <- switch(
    model_types,
    lin_interact = "train_lin_interact.R",
    lasso        = "train_lasso.R",
    dnn          = "train_dnn.R",
    spinn        = "train_spinn.R",
    spinn2       = "train_spinn2.R",
    xgboost      = "train_xgboost.R",
    rf           = "train_rf.R",    
    svm          = "train_svm.R",    
    stop(sprintf("Unsupported model_types '%s'", model_types))
  )

  script_paths <- list.files(
    path       = ".",
    pattern    = paste0(script_name, "$"),
    recursive  = TRUE,
    full.names = TRUE
  )
  if (length(script_paths) == 0) {
    stop(sprintf(
      "Cannot find training script '%s' in working dir or subdirs",
      script_name
    ))
  }
  source(script_paths[1])

  # 4. train the model
  model <- switch(
    model_types,
    lin_interact = train_lin_interact(X, y),
    lasso        = train_lasso(X, y),
    dnn          = train_dnn(X, y),
    spinn        = train_spinn(X, y),
    spinn2       = train_spinn2(X, y),
    xgboost      = train_xgboost(X, y),
    rf           = train_rf(X, y),    
    svm          = train_svm(X, y),   
    stop(sprintf("Unsupported model_types '%s'", model_types))
  )

  # 5. assign seeds, run interpreters
  method_seeds <- seed + seq_along(interpreters)
  metrics_list <- lapply(seq_along(interpreters), function(i) {
    method      <- interpreters[i]
    method_seed <- method_seeds[i]
    set.seed(method_seed)
    out <- explain_wrapper(
      X          = X,
      y          = y,
      model      = model,
      method     = method,
      K          = K,
      seed       = method_seed,
      active_idx = active_idx
    )
    selected <- out$selected
    rec      <- sum(selected %in% active_idx)
    recall_at_k    <- rec / length(active_idx)
    precision_at_k <- rec / K
    fp_at_k        <- K - rec
    f1 <- if ((recall_at_k + precision_at_k) > 0) {
      2 * recall_at_k * precision_at_k /
        (recall_at_k + precision_at_k)
    } else 0
    message(sprintf(
      "[DEBUG] repeat %3d | %-12s | seed=%5d | sel=%2d/%2d",
      repeat_id, method, method_seed, rec, length(active_idx)
    ))
    data.frame(
      repeat_id      = repeat_id,
      interpreter    = method,
      recall_at_k    = recall_at_k,
      precision_at_k = precision_at_k,
      fp_at_k        = fp_at_k,
      f1             = f1,
      stringsAsFactors = FALSE
    )
  })

  # 6. return interface
  list(table = bind_rows(metrics_list))
}