source("real_data/prep_adni_baseline.R")
source("real_data/plot_real_linear.R")

suppressPackageStartupMessages({
  library(Matrix)          # matrix utilities for safe numeric ops            
  library(glmnet)          # linear model for white-box extreme       
  library(fastshap)        # SHAP approximation                         
  library(future.apply)    # parallel map with RNG control                  
  library(data.table)      # .CSV export                                
  library(dplyr)           # tidy summaries                                
  library(readxl)          # Excel reader                                     
})

## 1) Assert preprocess functions exist
if (!exists("prep_adni_baseline", mode = "function"))
  stop("Missing function 'prep_adni_baseline'. Please source your ADNI prep function before running.")     
## 2) Utility: robust target resolution 
resolve_target_col <- function(df, target) {
  # Resolve the actual target name in 'df' given 'target' with case-insensitive and synonyms  
  if (target %in% names(df)) return(target)                                                  
  nms <- names(df)                                                                           
  idx_ci <- which(tolower(nms) == tolower(target))                                           
  if (length(idx_ci) == 1) return(nms[idx_ci])                                               
  synonyms <- c("MMSCORE", "MMSE", "MMSE_TOTAL", "MMSEScore", "MMSE.Total")                  
  idx_syn <- which(tolower(nms) %in% tolower(synonyms))                                      
  if (length(idx_syn) >= 1) return(nms[idx_syn[1]])                                          
  stop(sprintf("Target '%s' not found. Available columns (first 30): %s ...",
               target, paste(head(names(df), 30), collapse = ", ")))                        
}

## 3) Metrics (GLOBAL)
cosine_sim <- function(a, b, eps = 1e-12) {
  a <- as.numeric(a); b <- as.numeric(b)
  if (length(a) != length(b)) {
    k <- min(length(a), length(b)); a <- a[seq_len(k)]; b <- b[seq_len(k)]
  }
  a[!is.finite(a)] <- 0; b[!is.finite(b)] <- 0
  na <- sqrt(sum(a * a)); nb <- sqrt(sum(b * b))
  if (!is.finite(na) || !is.finite(nb) || na <= eps || nb <= eps) return(0)
  v <- sum(a * b) / (na * nb); max(-1, min(1, v))
}

# Number of strictly zero entries outside Top-K
sparsity_l0 <- function(attr, K = 50) {
  a <- abs(as.numeric(attr))
  if (!length(a)) return(NA_real_)
  k <- min(K, length(a)); if (k <= 0) return(NA_real_)
  sel <- order(a, decreasing = TRUE)[seq_len(k)]
  if (length(a) == k) return(0L)
  sum(a[-sel] == 0)
}

# Stability (avg. cosine similarity across seeds)
pairwise_mean_cos <- function(lst) {
  lst <- lapply(lst, sanitize_attr)
  if (length(lst) < 2) return(1)
  s <- 0; c <- 0L
  for (u in 1:(length(lst)-1)) {
    for (v in (u+1):length(lst)) {
      s <- s + cosine_sim(lst[[u]], lst[[v]])
      c <- c + 1L
    }
  }
  if (c == 0L) return(1)
  s / c
}

# Infidelity
#   attr_unit = "z":  attr in masking Z (BayesSLE/SHAP/LIME)
#   attr_unit = "x":  attr in input delta(x)(IG)
#   normalize = "fxfb":  (f(x)-f(baseline))^2 normalized 
infidelity_metric <- function(f_pred, x, baseline, attr, pert_masks,
                              seed = 1L,
                              attr_unit = c("z","x"),
                              normalize = c("fxfb","var","none","range"),
                              eps = 1e-12) {
  attr_unit <- match.arg(attr_unit)
  normalize <- match.arg(normalize)

  set.seed(seed)
  x <- as.numeric(x); 
  baseline <- as.numeric(baseline)
  Z <- matrix(as.integer(pert_masks), nrow = nrow(pert_masks))  # 1=baseline, 0=keep x

  # X'(Z) = (1-Z)*x + Z*baseline
  Xp <- sweep(1L - Z, 2L, x, "*") + sweep(Z, 2L, baseline, "*")

  fx  <- as.numeric(f_pred(matrix(x,  1L)))
  fxp <- as.numeric(f_pred(Xp))
  dfx <- fx - fxp

  # Linearization term
  if (attr_unit == "z") {
    ip <- as.numeric(- Z %*% as.numeric(attr))              
  } else {
    Dx <- sweep(matrix(rep(x, nrow(Xp)), nrow = nrow(Xp), byrow = TRUE), 2L, Xp, "-")
    ip <- as.numeric(Dx %*% as.numeric(attr))
  }
  mse <- mean((ip - dfx)^2)

  denom <- switch(normalize,
                  fxfb  = { fb <- as.numeric(f_pred(matrix(baseline, 1L))); (fx - fb)^2 },
                  var   = stats::var(dfx),
                  range = { rg <- diff(range(dfx)); rg*rg },
                  none  = 1)
  if (!is.finite(denom) || denom < eps) denom <- eps

  list(
    value    = if (normalize == "none") mse else mse/denom,
    mse      = mse,
    denom    = denom,
    dfx_mean = mean(dfx),
    dfx_sd   = stats::sd(dfx)
  )
}

## Models: Linear (ridge) 
train_linear_ridge <- function(X, y, seed = 2025, alpha = 0) {
  # Train ridge regression (alpha=0) for regression target; return predict + coef 
  set.seed(seed)                                                                  
  y <- as.numeric(y)                                                              
  cv <- cv.glmnet(X, y, family = "gaussian", alpha = alpha, nfolds = 5)          
  fit <- glmnet(X, y, family = "gaussian", alpha = alpha, lambda = cv$lambda.min)
  pred <- function(newX) as.numeric(glmnet::predict.glmnet(fit, newx = as.matrix(newX)))  
  coef_vec <- as.numeric(coef(fit))                                              
  list(predict = pred, coef = coef_vec, lambda = cv$lambda.min)                  
}

# 5) Explainers: BayesSLE wrapper 
train_spinn <- function(
  X, y,
  seed       = 42L,
  epochs     = 15L,
  batch_size = 64L,
  l1_input   = 1e-3,
  verbose    = 0L
) {
  if (!requireNamespace("keras3", quietly = TRUE))
    stop("Package 'keras3' is not installed.")
  if (!requireNamespace("tensorflow", quietly = TRUE))
    stop("Package 'tensorflow' is not installed.")

  tf <- tensorflow::tf

  set.seed(as.integer(seed))
  try(tf$random$set_seed(as.integer(seed)), silent = TRUE)
  try(tf$config$set_visible_devices(list(), "GPU"), silent = TRUE)
  try(tf$config$threading$set_intra_op_parallelism_threads(1L), silent = TRUE)
  try(tf$config$threading$set_inter_op_parallelism_threads(1L), silent = TRUE)

  X <- as.matrix(X); y <- as.numeric(y)
  p <- ncol(X); if (is.null(p) || p < 1L) stop("[SPINN] invalid X (no columns).")

  x_mean <- colMeans(X)
  x_sd   <- apply(X, 2, sd)
  if (any(x_sd == 0)) stop("[SPINN] constant column(s) in X; remove or jitter before training.")
  Xs <- scale(X, center = x_mean, scale = x_sd)

  y_mean <- mean(y)
  y_sd   <- sd(y)
  if (y_sd == 0) stop("[SPINN] y is constant; cannot train.")
  ys <- as.numeric((y - y_mean) / y_sd)

  # first Dense has weight matrix W1 with nrow = p
u1 <- as.integer(min(64L, max(8L, p)))
u2 <- as.integer(min(32L, max(4L, p %/% 2L)))

inputs  <- keras3::layer_input(shape = p)
h1 <- inputs |>
  keras3::layer_dense(
    units = u1, activation = "relu",
    kernel_regularizer = keras3::regularizer_l1(l1_input)
  )
h2 <- h1 |>
  keras3::layer_dense(units = u2, activation = "relu")
outputs <- h2 |>
  keras3::layer_dense(units = 1L, activation = "linear")

model <- keras3::keras_model(inputs = inputs, outputs = outputs)

keras3::compile(
  model,
  optimizer = keras3::optimizer_adam(learning_rate = 1e-3),
  loss      = "mse",
  metrics   = list("mse")
)

  cb <- list(
    keras3::callback_early_stopping(
      monitor = "val_loss", patience = 3L, restore_best_weights = TRUE
    )
  )

  keras3::fit(
    model,
    x = Xs, y = ys,
    validation_split = 0.15,
    epochs           = as.integer(epochs),
    batch_size       = as.integer(batch_size),
    callbacks        = cb,
    verbose          = as.integer(verbose),
    shuffle          = TRUE
  )

  py_obj_to_matrix <- function(obj) {
    if (is.null(obj)) return(NULL)
    mat <- tryCatch({
      np <- obj$numpy(); as.matrix(reticulate::py_to_r(np))
    }, error = function(e) NULL)
    if (!is.null(mat)) return(mat)
    tryCatch(as.matrix(reticulate::py_to_r(obj)), error = function(e) NULL)
  }

  W1 <- NULL
  if (!is.null(model$layers) && length(model$layers) > 0) {
    for (k in seq_along(model$layers)) {
      lyr <- model$layers[[k]]
      W_try <- tryCatch(py_obj_to_matrix(lyr$kernel), error = function(e) NULL)
      if (is.null(W_try)) {
        W_try <- tryCatch({
          ws <- lyr$weights
          if (!is.null(ws) && length(ws) >= 1) py_obj_to_matrix(ws[[1]]) else NULL
        }, error = function(e) NULL)
      }
      if (!is.null(W_try) && length(dim(W_try)) == 2) {
        if (nrow(W_try) == p || ncol(W_try) == p) {
          if (nrow(W_try) != p && ncol(W_try) == p) W_try <- t(W_try)
          if (nrow(W_try) == p) { W1 <- W_try; break }
        }
      }
    }
  }
  if (is.null(W1)) {
    W1 <- tryCatch({
      tv <- model$trainable_variables
      if (!is.null(tv) && length(tv) >= 1) py_obj_to_matrix(tv[[1]]) else NULL
    }, error = function(e) NULL)
    if (!is.null(W1) && length(dim(W1)) == 2 && nrow(W1) != p && ncol(W1) == p) W1 <- t(W1)
  }
  if (is.null(W1) || length(dim(W1)) != 2 || nrow(W1) != p)
    stop("[SPINN] Could not extract first Dense kernel with input dimension p.")

  s_raw <- rowSums(abs(W1))
  rng   <- max(s_raw) - min(s_raw)
  if (!is.finite(rng) || rng == 0)
    stop("[SPINN] First-layer kernel produced constant scores; tune l1_input/capacity.")
  s_norm <- (s_raw - min(s_raw)) / rng

  list(
    model   = model,
    x_mean  = x_mean,
    x_sd    = x_sd,
    y_mean  = y_mean,
    y_sd    = y_sd,
    scores  = as.numeric(s_norm)  
  )
}

# Extract global structure scores s_j from a trained SPINN.
# We aggregate absolute weights in the first layer across hidden units, then min-max normalize to [0,1].
spinn_feature_scores <- function(spinn_fit) {
  if (!is.list(spinn_fit) || is.null(spinn_fit$model))
    stop("spinn_feature_scores(): 'spinn_fit' must be the list returned by train_spinn().")

  if (!is.null(spinn_fit$scores)) {
    s <- as.numeric(spinn_fit$scores)
    p <- length(spinn_fit$x_mean)
    if (length(s) != p) stop("spinn_feature_scores(): cached scores length mismatch.")
    return(s)
  }

  mdl <- spinn_fit$model
  p   <- length(spinn_fit$x_mean)

  py_obj_to_matrix <- function(obj) {
    if (is.null(obj)) return(NULL)
    mat <- tryCatch({
      np <- obj$numpy(); as.matrix(reticulate::py_to_r(np))
    }, error = function(e) NULL)
    if (!is.null(mat)) return(mat)
    tryCatch(as.matrix(reticulate::py_to_r(obj)), error = function(e) NULL)
  }

  extract_W_from_layer <- function(lyr) {
    W <- tryCatch(py_obj_to_matrix(lyr$kernel), error = function(e) NULL)
    if (!is.null(W)) return(W)
    W <- tryCatch({
      ws <- lyr$weights
      if (!is.null(ws) && length(ws) >= 1) py_obj_to_matrix(ws[[1]]) else NULL
    }, error = function(e) NULL)
    if (!is.null(W)) return(W)
    tryCatch({
      gw <- lyr$get_weights
      if (is.function(gw)) {
        ww <- reticulate::py_to_r(gw()); as.matrix(ww[[1]])
      } else NULL
    }, error = function(e) NULL)
  }

  W1 <- NULL
  if (!is.null(mdl$layers) && length(mdl$layers) > 0) {
    for (k in seq_along(mdl$layers)) {
      lyr <- mdl$layers[[k]]
      W_try <- extract_W_from_layer(lyr)
      if (!is.null(W_try) && length(dim(W_try)) == 2) {
        if (nrow(W_try) == p || ncol(W_try) == p) {
          if (nrow(W_try) != p && ncol(W_try) == p) W_try <- t(W_try)
          if (nrow(W_try) == p) { W1 <- W_try; break }
        }
      }
    }
  }
  if (is.null(W1)) {
    W1 <- tryCatch({
      tv <- mdl$trainable_variables
      if (!is.null(tv) && length(tv) >= 1) py_obj_to_matrix(tv[[1]]) else NULL
    }, error = function(e) NULL)
    if (!is.null(W1) && length(dim(W1)) == 2 && nrow(W1) != p && ncol(W1) == p) W1 <- t(W1)
  }
  if (is.null(W1) || length(dim(W1)) != 2 || nrow(W1) != p)
    stop("spinn_feature_scores(): could not extract first Dense kernel with input dimension p.")

  s_raw <- sqrt(rowSums(W1^2)) 
  rng   <- max(s_raw) - min(s_raw)
  if (!is.finite(rng) || rng == 0)
    stop("SPINN produced constant feature scores; consider increasing l1_input or capacity.")
  (s_raw - min(s_raw)) / rng
}

# Wrap a fitted SPINN into a function compatible with explain_bayes_sle
spinn_as_score_fn <- function(spinn_fit) {
  function(X, y) spinn_feature_scores(spinn_fit)
}

explain_bayes_sle <- function(
  X, y, model,
  K = 50,
  seed = 123,
  spinn_scores,            
  n_samples = 2000,
  stan_iter = 1000, stan_chains = 2,
  use_vi = TRUE,
  active_idx = NULL,
  x_index
) { 
  # helpers
  as_mat <- function(M) {
    M <- as.matrix(M)
    if (is.null(colnames(M))) colnames(M) <- paste0("V", seq_len(ncol(M)))
    M
  }

  # Universal numeric predictor for common black-box models
  predict_blackbox_numeric <- function(mdl, Xnew) {
    Xnew <- as.matrix(Xnew)
    if (is.function(mdl)) {
      yh <- mdl(Xnew)
    } else if (inherits(mdl, "xgb.Booster")) {
      yh <- predict(mdl, Xnew)
    } else if (inherits(mdl, "randomForest")) {
      yh <- predict(mdl, newdata = as.data.frame(Xnew))
    } else if ("ranger" %in% class(mdl)) {
      yh <- predict(mdl, data = as.data.frame(Xnew))$predictions
    } else if (inherits(mdl, "cv.glmnet")) {

  yh <- as.numeric(glmnet::predict.glmnet(mdl$glmnet.fit, newx = Xnew, s = mdl$lambda.min))
} else if (inherits(mdl, "glmnet")) {
  yh <- as.numeric(glmnet::predict.glmnet(mdl, newx = Xnew))
    } else if (inherits(mdl, "svm")) {
      yh <- predict(mdl, Xnew)
    } else if (inherits(mdl, c("keras.src.models.sequential.Sequential","keras.src.models.functional.Functional"))) {
      yh <- mdl$predict(Xnew, verbose = 0L)
    } else {
      stop("Unsupported model type for prediction.")
    }
    yh <- as.numeric(yh); if (is.matrix(yh)) yh <- yh[,1]; as.numeric(yh)
  }

  # Get SPINN structure scores s_j in [0,1]
get_spinn_scores <- function(spinn_model, X, y, p) {
  if (!is.list(spinn_model) || is.null(spinn_model$model))
    stop("get_spinn_scores(): 'spinn_model' must be the object returned by train_spinn().")
  s <- spinn_feature_scores(spinn_model)
  s <- as.numeric(s)
  if (length(s) != p)
    stop(sprintf("SPINN scores length mismatch: got %d, expected %d (= ncol(X)).", length(s), p))
  rng <- max(s) - min(s)
  if (!is.finite(rng) || rng <= 0)
    stop("SPINN scores are constant or non-finite.")
  (s - min(s)) / rng
}

  # Cosine distance between rows of A and vector b: d = 1 - cos(theta)
  cosine_dist_to <- function(A, b) {
    nb <- sqrt(sum(b^2))
    if (nb == 0) stop("x0 has zero L2 norm; cannot compute cosine distance.")
    num <- as.numeric(A %*% b)
    na  <- sqrt(rowSums(A^2))
    cosv <- num / (na * nb)
    cosv[cosv >  1] <-  1
    cosv[cosv < -1] <- -1
    1 - cosv
  }

  # Compile Stan model
  compile_stan_model <- function(code) {
    use_cmdstanr <- requireNamespace("cmdstanr", quietly = TRUE)
    if (use_cmdstanr) {
      return(list(
        backend = "cmdstanr",
        mod = cmdstanr::cmdstan_model(cmdstanr::write_stan_file(code))
      ))
    }
    if (requireNamespace("rstan", quietly = TRUE)) {
      return(list(
        backend = "rstan",
        mod = rstan::stan_model(model_code = code)
      ))
    }
    stop("Neither cmdstanr nor rstan is available.")
  }

  stan_fit_and_draws <- function(mod, backend, data_list, use_vi, stan_iter, stan_chains, seed) {
  if (backend == "cmdstanr") {
    if (use_vi) {
      fit <- mod$variational(
        data = data_list, seed = seed, iter = 8000,
        algorithm = "fullrank", eta = 0.05, grad_samples = 8,
        elbo_samples = 20, tol_rel_obj = 0.001
      )
    } else {
      fit <- mod$sample(
        data = data_list, seed = seed, chains = stan_chains,
        iter_warmup = ceiling(stan_iter/2),
        iter_sampling = floor(stan_iter/2),
        refresh = 0
      )
    }
    drw <- fit$draws(c("beta","alpha","sigma"))
    if (requireNamespace("posterior", quietly = TRUE)) {
      draws_mat <- posterior::as_draws_matrix(drw)
      draws_mat <- as.matrix(draws_mat)
    } else {
      draws_mat <- as.matrix(drw)
    }
  } else if (backend == "rstan") {
    if (use_vi) {
      fit <- rstan::vb(mod, data = data_list, seed = seed, iter = 8000, eta = 0.05)
    } else {
      fit <- rstan::sampling(mod, data = data_list, seed = seed,
                             iter = stan_iter, warmup = floor(stan_iter/2),
                             chains = stan_chains, refresh = 0)
    }
    draws_mat <- as.matrix(fit, pars = c("beta","alpha","sigma"))
  } else {
    stop("Unknown Stan backend.")
  }

  cn  <- colnames(draws_mat)
  idb <- grep("^beta(\\[|\\.|$)", cn)
  ida <- which(cn == "alpha")
  ids <- which(cn == "sigma")
  if (!length(idb) || !length(ida) || !length(ids))
    stop("Failed to extract beta/alpha/sigma draws from Stan fit.")

  beta_mat  <- as.matrix(draws_mat[, idb, drop = FALSE])
  alpha_vec <- as.numeric(draws_mat[, ida])
  sigma_vec <- as.numeric(draws_mat[, ids])

  list(beta = beta_mat, alpha = alpha_vec, sigma = sigma_vec)
}

  # Compute WAIC given draws on y_std scale with heteroskedastic noise
  waic_from_draws <- function(Zstd, y_std, w, draws) {
  S <- nrow(draws$beta); N <- nrow(Zstd); P <- ncol(Zstd)
  if (ncol(draws$beta) != P) stop("Draws beta dimension mismatch.")

  mu <- matrix(0.0, nrow = N, ncol = S)
  for (s in seq_len(S)) {
    beta_s <- as.numeric(draws$beta[s, ])  
    mu[, s] <- draws$alpha[s] + as.numeric(Zstd %*% beta_s)
  }

  sig2 <- matrix(rep(as.numeric(draws$sigma)^2, each = N) / as.numeric(w), nrow = N, ncol = S)
  ll   <- -0.5*log(2*pi) - 0.5*log(sig2) - ((y_std - mu)^2)/(2*sig2)

  lppd  <- sum(log(rowMeans(exp(ll))))
  pwaic <- sum(apply(ll, 1, stats::var))
  waic  <- -2*(lppd - pwaic)
  as.numeric(waic)
}

  # inputs
  X <- as_mat(X)
  y <- as.numeric(y)
  if (missing(x_index) || length(x_index) != 1L || !(x_index %in% seq_len(nrow(X))))
    stop("x_index must be provided and in [1, nrow(X)].")

  p  <- ncol(X)
  if (missing(spinn_scores) || !is.numeric(spinn_scores) || length(spinn_scores) != p)
    stop("explain_bayes_sle(): 'spinn_scores' must be a numeric vector of length ncol(X).")

  x0 <- as.numeric(X[x_index, , drop = TRUE])
  b  <- colMeans(X)
  Delta <- b - x0

  # structure-aware masks Z
  set.seed(seed)
  s_raw <- as.numeric(spinn_scores)
  rng   <- max(s_raw) - min(s_raw)
  if (!is.finite(rng) || rng <= 0) stop("SPINN scores are constant or non-finite.")
  s <- (s_raw - min(s_raw)) / rng           # normalized to [0,1]
  rho_min <- 0.1; rho_max <- 0.9
  rho <- rho_min + (rho_max - rho_min) * (1 - s)
  Z <- matrix(rbinom(n_samples * p, 1L, rep(rho, each = n_samples)), nrow = n_samples, ncol = p)

  # Perturbed inputs and response deltas
  Xpert <- sweep(1L - Z, 2L, x0, "*") + sweep(Z, 2L, b, "*")
  f_pred <- function(newX) predict_blackbox_numeric(model, newX)
  fx  <- f_pred(matrix(x0, nrow = 1L))
  y_p <- f_pred(Xpert)
  dy  <- y_p - fx
  if (stats::var(dy) == 0) stop("Variance of dy is zero; choose a different baseline or sampling.")

  # cosine kernel weights
  # Cosine distance d in [0,2]; Gaussian kernel K(d;lambda)=exp(-d^2/(2*lambda^2))
  dist <- cosine_dist_to(Xpert, x0)
  r_eff <- stats::median(dist[dist > 0])
  if (!is.finite(r_eff) || r_eff <= 0) stop("Effective radius r could not be computed (all distances zero).")
  lambda <- r_eff / sqrt(2)
  w <- exp(-(dist^2) / (2 * lambda^2))

  # Regressors: mask features
  DZ <- sweep(Z, 2L, Delta, "*")

  # weighted EN screening
  en_cv <- glmnet::cv.glmnet(
    x = DZ, y = dy, weights = w,
    alpha = 0.5, intercept = TRUE, standardize = TRUE,
    nfolds = 5, type.measure = "mse"
  )
  imp <- abs(as.numeric(stats::coef(en_cv, s = "lambda.min"))[-1])
  imp[is.na(imp)] <- 0
  K_scr <- min(K, length(imp))
  if (K_scr < 1) stop("Screening produced zero variables.")
  topk_vars <- head(order(imp, decreasing = TRUE), K_scr)

  # weighted standardization in the selected subspace
  Zsel   <- DZ[, topk_vars, drop = FALSE]
  wbar   <- sum(w)
  z_mean <- as.numeric(colSums(w * Zsel) / wbar)
  Zc     <- sweep(Zsel, 2L, z_mean, "-")
  z_sd   <- sqrt(colSums(w * (Zc^2)) / wbar)
  if (any(z_sd == 0)) stop("Selected features contain zero variance under weights.")
  Zstd   <- sweep(Zc, 2L, z_sd, "/")
  y_mean <- sum(w * dy) / wbar
  y_std  <- dy - y_mean

  # Stan model
  stan_code <- "
  data {
    int<lower=1> N;
    int<lower=1> P;
    matrix[N,P] Z;
    vector[N] y;
    vector<lower=0>[N] w;
    real<lower=0> lambda;
  }
  parameters {
    real alpha;
    vector[P] beta;
    real<lower=0> sigma;
  }
  model {
    alpha ~ normal(0, 5);
    beta  ~ double_exponential(0, 1/lambda); // Bayesian Lasso
    sigma ~ student_t(3, 0, 2.5);
    for (n in 1:N)
      y[n] ~ normal(alpha + Z[n] * beta, sigma / sqrt(w[n]));
  }"

  sb <- compile_stan_model(stan_code)

  lam0 <- sqrt(log(ncol(Zstd)))
  lambda_grid <- lam0 * c(0.5, 1.0, 2.0)

  best <- NULL; best_waic <- Inf

  for (lam in lambda_grid) {
    stan_data <- list(
      N = nrow(Zstd), P = ncol(Zstd), Z = Zstd,
      y = as.numeric(y_std), w = as.numeric(w), lambda = lam
    )

    draws <- stan_fit_and_draws(sb$mod, sb$backend, stan_data, use_vi, stan_iter, stan_chains, seed)
    # WAIC selection (no fallback)
    waic <- waic_from_draws(Zstd, y_std, w, draws)

    if (waic < best_waic) {
      best_waic <- waic
      # Posterior summaries on standardized scale
      beta_mean_std <- colMeans(draws$beta)
      beta_sd_std   <- apply(draws$beta, 2, stats::sd)
      alpha_mean    <- mean(draws$alpha)

      # Map back to unstandardized DZ-scale phi_x
      phi_x_mean <- beta_mean_std / z_sd
      phi_x_sd   <- beta_sd_std   / z_sd

      phi_x_full_mean <- phi_x_full_sd <- numeric(p)
      phi_x_full_mean[topk_vars] <- phi_x_mean
      phi_x_full_sd[topk_vars]   <- phi_x_sd

      contrib_z <- phi_x_full_mean * Delta

      best <- list(
        phi_x_full_mean = phi_x_full_mean,
        phi_x_full_sd   = phi_x_full_sd,
        contrib_z       = contrib_z,
        topk_vars       = topk_vars,
        lambda          = lam,
        alpha_mean      = alpha_mean
      )
    }
  }

  if (is.null(best)) stop("No valid model was selected by WAIC.")

  # output
  out_df <- data.frame(
    variable = colnames(X),
    mean     = as.numeric(best$contrib_z),                  # contribution on Z-scale
    sd       = as.numeric(abs(Delta) * best$phi_x_full_sd) 
  )

  list(
    topk_vars   = best$topk_vars,
    posterior   = out_df,
    best_lambda = best$lambda,
    spinn_score = s,              
    x0_index    = x_index,
    attr_x      = as.numeric(best$phi_x_full_mean),
    attr_z      = as.numeric(best$contrib_z)
  )
}

run_bayesSLE <- function(X, y, f_pred, x_index, K = 50, seed = 1,
                         n_samples = 2000, stan_iter = 1000, stan_chains = 2,
                         spinn_scores, use_vi = FALSE) {  
  t0 <- proc.time()[3]

  out <- explain_bayes_sle(
    X = X, y = y, model = f_pred,
    K = K, seed = seed, spinn_scores = spinn_scores,   
    n_samples = n_samples, stan_iter = stan_iter,
    stan_chains = stan_chains, use_vi = use_vi,
    active_idx = NULL,
    x_index = x_index
  )
  rt <- proc.time()[3] - t0

  post <- out$posterior
  if (!("mean" %in% names(post))) stop("BayesSLE returned posterior without 'mean' column.")
  if (!("variable" %in% names(post))) stop("BayesSLE returned posterior without 'variable' column.")

  p <- ncol(X)
  if (nrow(post) < p) {
    stop("Posterior length (rows) is less than number of predictors.")
  }

  x_cols <- colnames(X)
  if (is.null(x_cols)) x_cols <- paste0("V", seq_len(p))
  if (!setequal(x_cols, post$variable)) {
    stop("Posterior 'variable' names do not match X column names.")
  }
  idx <- match(x_cols, post$variable)
  if (anyNA(idx)) stop("Failed to align posterior to X by variable names.")

  att <- as.numeric(post$mean)[idx]
  if (!all(is.finite(att))) {
    bad <- which(!is.finite(att))
    stop(sprintf("Non-finite attributions at positions: %s", paste(bad, collapse = ",")))
  }

  list(
    attr = att,
    runtime = rt,
    topk_vars = if (!is.null(out$topk_vars)) as.integer(out$topk_vars) else integer(0)
  )
}

run_SHAP <- function(X_train, f_pred, x, m = 2048, seed = 1) {
  t0 <- proc.time()[3]
  x <- as.numeric(x); p <- length(x)
  if (p != ncol(X_train)) stop("KernelSHAP: length(x) must equal ncol(X_train).")
  baseline <- colMeans(X_train)

  set.seed(seed)
  # 1 = baseline, 0 = keep x
  Z <- matrix(rbinom(m * p, 1L, 0.5), nrow = m, ncol = p)
  Z <- unique(rbind(Z, rep(0L, p), rep(1L, p)))
  M <- nrow(Z)

  # X'(Z) = (1-Z)*x + Z*baseline
  Xz <- sweep(1L - Z, 2L, x, "*") + sweep(Z, 2L, baseline, "*")

  y_raw <- as.numeric(f_pred(Xz))
  if (length(y_raw) != M)
    stop(sprintf("KernelSHAP: predictor returned %d for %d inputs.", length(y_raw), M))
  fb <- as.numeric(f_pred(matrix(baseline, nrow = 1)))
  fx <- as.numeric(f_pred(matrix(x, nrow = 1)))
  y  <- y_raw - fb

  # Shapley kernel
  k <- rowSums(Z)
  w <- numeric(M)
  idx_mid <- which(k > 0 & k < p)
  if (length(idx_mid)) {
    logw <- log(p - 1) - lchoose(p, k[idx_mid]) - log(k[idx_mid]) - log(p - k[idx_mid])
    w[idx_mid] <- exp(logw)
  }

  W_ENDPOINT <- if (length(idx_mid)) max(w[idx_mid]) * 1e3 else 1e3
  w[k == 0 | k == p] <- W_ENDPOINT
  if (any(!is.finite(w)) || any(w <= 0)) stop("KernelSHAP: invalid weights encountered.")

  D <- 1 - Z
  D <- as.matrix(D) * 1.0

  sw  <- sqrt(w)
  Xw  <- D * sw
  yw  <- y * sw
  XtX <- crossprod(Xw)          # D' W D
  Xty <- crossprod(Xw, yw)      # D' W y

  lam <- 1e-8 * mean(diag(XtX))
  if (!is.finite(lam) || lam <= 0) lam <- 1e-8
  phi <- as.numeric(solve(XtX + diag(lam, ncol(XtX)), Xty))

  add_err <- abs((fx - fb) - sum(phi))
  list(attr = phi, runtime = proc.time()[3] - t0, additivity_error = add_err)
}

run_lime <- function(X_train, f_pred, x,
                     n_features = 50, n_permutations = 600,
                     sparsity = 2, seed = 1) {
  t0 <- proc.time()[3]
  set.seed(seed)

  x0 <- as.numeric(x); p <- length(x0)
  if (p != ncol(X_train)) stop("LIME: length(x) must equal ncol(X_train).")
  baseline <- colMeans(X_train)

  Z <- matrix(0L, nrow = n_permutations, ncol = p)
  if (sparsity > 0) {
    for (j in seq_len(n_permutations)) {
      idx <- if (sparsity >= p) seq_len(p) else sample.int(p, sparsity)
      Z[j, idx] <- 1L
    }
  }
  Z <- unique(rbind(Z, rep(0L, p), rep(1L, p)))
  m <- nrow(Z)

  Xpert <- sweep(1L - Z, 2L, x0, "*") + sweep(Z, 2L, baseline, "*")
  y <- as.numeric(f_pred(Xpert))
  if (length(y) != m) stop(sprintf("LIME: predictor returned %d for %d inputs.", length(y), m))

  # LIME weighting
  mu  <- colMeans(X_train)
  sdv <- apply(X_train, 2, sd)
  sdv[!is.finite(sdv) | sdv == 0] <- 1

  x0_std    <- (x0 - mu) / sdv
  Xpert_std <- sweep(Xpert, 2L, mu, "-")
  Xpert_std <- sweep(Xpert_std, 2L, sdv, "/")

  d <- sqrt(rowSums((Xpert_std - matrix(x0_std, nrow = m, ncol = p, byrow = TRUE))^2))
  kernel_width <- sqrt(p)
  w <- exp(-(d^2) / (kernel_width^2))

  if (any(!is.finite(w)) || any(w <= 0)) stop("LIME: invalid kernel weights.")

  cvfit <- glmnet::cv.glmnet(
    x = Z, y = y, weights = w,
    alpha = 1, intercept = TRUE, standardize = TRUE,
    nfolds = 5, type.measure = "mse"
  )
  a <- as.numeric(stats::coef(cvfit, s = "lambda.min"))[-1]
  if (any(is.na(a))) stop("LIME: NA coefficients from glmnet; check design/weights.")

  # only retain Top-K, according to |a|
  K <- min(n_features, p)
  if (K < p) {
    keep <- order(abs(a), decreasing = TRUE)[seq_len(K)]
    mask <- logical(p); mask[keep] <- TRUE
    a[!mask] <- 0
  }

  list(attr = as.numeric(a), runtime = proc.time()[3] - t0)
}

run_ig <- function(model, x, baseline, steps = 50, linear_beta = NULL, std_fn = NULL) {
  t0 <- proc.time()[3]
  x <- as.numeric(x); baseline <- as.numeric(baseline)
  if (length(x) != length(baseline)) stop("IG: x and baseline must have the same length.")

  if (!is.null(linear_beta)) {
    beta <- as.numeric(linear_beta)
    if (length(beta) != (length(x) + 1)) stop("IG: linear_beta should include intercept and match p+1.")
    attr <- (x - baseline) * beta[-1]
    return(list(attr = attr, runtime = proc.time()[3] - t0))
  }

  if (is.null(model)) stop("IG: 'model' must be a keras model when linear_beta is NULL.")
  if (is.null(std_fn)) stop("IG: 'std_fn' must be provided to standardize inputs for the DNN.")
  if (!requireNamespace("tensorflow", quietly = TRUE))
    stop("IG: TensorFlow is required.")

  tf <- tensorflow::tf

  # Standardize single examples to the DNN's expected scale
  x_s <- std_fn(matrix(x, nrow = 1L))
  b_s <- std_fn(matrix(baseline, nrow = 1L))

  x_t <- tf$cast(tf$constant(x_s), dtype = "float32")
  b_t <- tf$cast(tf$constant(b_s), dtype = "float32")

  y_test <- model(x_t)
  shp <- as.integer(y_test$shape$as_list())
  if (!(length(shp) %in% c(1,2) && prod(shp) == 1L)) {
    stop("IG: model must return a scalar for the given input. Select a target output explicitly.")
  }

  alphas <- tf$linspace(tf$constant(0, dtype = "float32"),
                        tf$constant(1, dtype = "float32"), as.integer(steps))
  total_grad <- tf$zeros_like(x_t)

  for (i in seq_len(steps)) {
    a <- alphas[i]
    z <- b_t + a * (x_t - b_t)
    with(tf$GradientTape() %as% tape, {
      tape$watch(z)
      y_hat <- model(z)          # scalar
    })
    g <- tape$gradient(y_hat, z) # gradient wrt input
    total_grad <- total_grad + g
  }

  avg_grad <- total_grad / tf$cast(as.integer(steps), dtype = "float32")
  delta    <- x_t - b_t
  attr     <- as.numeric(reticulate::py_to_r((delta * avg_grad)$numpy()))

  list(attr = attr, runtime = proc.time()[3] - t0)
}


## 6) run_setting()
run_setting <- function(X_train, y_train, X_test, y_test,
                        model_type = c("linear"),
                        K = 50, seeds = 1:3,
                        n_points = 20, nsim_KernelSHAP = 300,
                        n_perms_lime = 600,
                        n_samples_bayes = 2000,
                        stan_iter = 1000, stan_chains = 2,
                        use_vi = FALSE,
                        spinn_scores) { 
  model_type <- match.arg(model_type)
  set.seed(2025)
  n_points  <- min(n_points, nrow(X_test))
  idx_points <- sample.int(nrow(X_test), n_points)

  # train model 
  if (model_type == "linear") {
    lin <- train_linear_ridge(X_train, y_train, seed = 2025, alpha = 0)
    f_pred      <- lin$predict
    baseline    <- colMeans(X_train)
    linear_beta <- lin$coef
  }

  # helpers 
  make_masks <- function(p, m = 512, keep_floor = 0.3, keep_cap = 0.7) {
  probs <- runif(p, keep_floor, keep_cap)
  matrix(rbinom(m * p, 1L, rep(probs, each = m)), nrow = m, ncol = p)
}

  sanitize_attr <- function(v) { v <- as.numeric(v); v[!is.finite(v)] <- 0; v }
  cosine_sim <- function(a, b, eps = 1e-12) {
    a <- as.numeric(a); b <- as.numeric(b)
    if (length(a) != length(b)) { k <- min(length(a), length(b)); a <- a[seq_len(k)]; b <- b[seq_len(k)] }
    a[!is.finite(a)] <- 0; b[!is.finite(b)] <- 0
    na <- sqrt(sum(a * a)); nb <- sqrt(sum(b * b))
    if (!is.finite(na) || !is.finite(nb) || na <= eps || nb <= eps) return(0)
    v <- sum(a * b) / (na * nb); max(-1, min(1, v))
  }

cosine_sim <- function(a, b, eps = 1e-6) {
  a <- as.numeric(a); b <- as.numeric(b)

  if (length(a) != length(b)) {
    k <- min(length(a), length(b))
    a <- a[seq_len(k)]; b <- b[seq_len(k)]
  }

  a[!is.finite(a)] <- 0
  b[!is.finite(b)] <- 0

  na <- sqrt(sum(a * a))
  nb <- sqrt(sum(b * b))
  if (!is.finite(na) || !is.finite(nb) || na <= eps || nb <= eps) return(0)

  v <- sum(a * b) / (na * nb)
  v <- max(-1, min(1, v))
  v
}

# Sparsity: #zeros outside Top-K by |attr|
sparsity_l0 <- function(attr, K = 50) {
  a <- abs(as.numeric(attr))
  if (!length(a)) return(NA_real_)
  k <- min(K, length(a))
  if (k <= 0) return(NA_real_)
  sel <- order(a, decreasing = TRUE)[seq_len(k)]
  if (length(a) == k) return(0L)
  sum(a[-sel] == 0)
}
  # parallel
  old_plan <- future::plan()
on.exit(future::plan(old_plan), add = TRUE)
future::plan(future::multisession, workers = max(1L, parallel::detectCores() - 1L))

  # per-point evaluation
  per_point <- future_lapply(
    seq_along(idx_points),
    function(ii) {
      i <- idx_points[ii]
      x  <- X_test[i, , drop = FALSE]
      cn <- colnames(X_test)
      pert_masks <- make_masks(ncol(X_test), m = 512)

      rows_metrics  <- list()
      rows_selected <- list()
      attr_store <- list(BayesSLE = list(), KernelSHAP = list(), LIME = list(), IG = list())

       for (sd in seeds) {
    bsl <- run_bayesSLE(
      X = X_test, y = y_test, f_pred = f_pred, x_index = i, K = K, seed = sd,
      n_samples = n_samples_bayes, stan_iter = stan_iter, stan_chains = stan_chains,
      spinn_scores = spinn_scores,     
      use_vi = use_vi
    )
        shp <- run_SHAP(X_train, f_pred, x = x, m = nsim_KernelSHAP, seed = sd)
        lme <- run_lime(X_train, f_pred, x = x, n_features = K,
                        n_permutations = n_perms_lime, seed = sd)
        ig <- run_ig(model = NULL, x = as.numeric(x), baseline = baseline,
             steps = 50, linear_beta = linear_beta, std_fn = NULL)
        a_bsl <- sanitize_attr(bsl$attr)
        a_shp <- sanitize_attr(shp$attr)
        a_lme <- sanitize_attr(lme$attr)
        a_ig  <- sanitize_attr(ig$attr)

        attr_store$BayesSLE[[as.character(sd)]]   <- a_bsl
        attr_store$KernelSHAP[[as.character(sd)]] <- a_shp
        attr_store$LIME[[as.character(sd)]]       <- a_lme
        attr_store$IG[[as.character(sd)]]         <- a_ig

fwrap <- function(newX) f_pred(newX)

ib     <- infidelity_metric(fwrap, x, baseline, a_bsl, pert_masks, sd, "z", "fxfb")
is     <- infidelity_metric(fwrap, x, baseline, a_shp, pert_masks, sd, "z", "fxfb")
il     <- infidelity_metric(fwrap, x, baseline, a_lme, pert_masks, sd, "z", "fxfb")
ii     <- infidelity_metric(fwrap, x, baseline, a_ig,  pert_masks, sd, "x", "fxfb")

ib_var <- infidelity_metric(fwrap, x, baseline, a_bsl, pert_masks, sd, "z", "var")$value
is_var <- infidelity_metric(fwrap, x, baseline, a_shp, pert_masks, sd, "z", "var")$value
il_var <- infidelity_metric(fwrap, x, baseline, a_lme, pert_masks, sd, "z", "var")$value
ii_var <- infidelity_metric(fwrap, x, baseline, a_ig,  pert_masks, sd, "x", "var")$value

ib_raw <- ib$mse; is_raw <- is$mse; il_raw <- il$mse; ii_raw <- ii$mse
r2     <- function(v) 1 - v

rows_metrics[[length(rows_metrics) + 1]] <- data.frame(
  point = i, seed = sd, model = model_type,
  method = c("BayesSLE","KernelSHAP","LIME","IG"),
  infidelity      = c(ib$value, is$value, il$value, ii$value),      
  infidelity_var  = c(ib_var, is_var, il_var, ii_var),              
  infidelity_raw  = c(ib_raw, is_raw, il_raw, ii_raw),              
  fidelity_R2     = r2(c(ib_var, is_var, il_var, ii_var)),          
  sparsity_topk_zeros = c(
    sparsity_l0(a_bsl, K = K),
    sparsity_l0(a_shp, K = K),
    sparsity_l0(a_lme, K = K),
    sparsity_l0(a_ig,  K = K)
  ),
  runtime_sec = c(bsl$runtime, shp$runtime, lme$runtime, ig$runtime)
)

        Kp <- min(K, length(a_bsl))
        pick <- function(a) head(order(abs(a), decreasing = TRUE), Kp)
        picks <- list(BayesSLE = pick(a_bsl), KernelSHAP = pick(a_shp),
                      LIME = pick(a_lme), IG = pick(a_ig))
        wmap  <- list(BayesSLE = a_bsl, KernelSHAP = a_shp, LIME = a_lme, IG = a_ig)

        for (mtd in names(picks)) {
          idx <- as.integer(picks[[mtd]])
          if (length(idx)) {
            rows_selected[[length(rows_selected) + 1]] <- data.frame(
              point  = i,
              seed   = sd,
              model  = model_type,
              method = mtd,
              rank   = seq_along(idx),
              var    = cn[idx],
              weight = wmap[[mtd]][idx]
            )
          }
        }

        if (!is.null(bsl$topk_vars)) {
          idx <- as.integer(bsl$topk_vars)
          if (length(idx)) {
            rows_selected[[length(rows_selected) + 1]] <- data.frame(
              point  = i,
              seed   = sd,
              model  = model_type,
              method = "BayesSLE_screen",
              rank   = seq_along(idx),
              var    = cn[idx],
              weight = NA_real_
            )
          }
        }
      } 

      stab_rows <- rbind(
        data.frame(point = i, model = model_type, method = "BayesSLE",
                   stability_cos = pairwise_mean_cos(attr_store$BayesSLE)),
        data.frame(point = i, model = model_type, method = "KernelSHAP",
                   stability_cos = pairwise_mean_cos(attr_store$KernelSHAP)),
        data.frame(point = i, model = model_type, method = "LIME",
                   stability_cos = pairwise_mean_cos(attr_store$LIME)),
        data.frame(point = i, model = model_type, method = "IG",
                   stability_cos = pairwise_mean_cos(attr_store$IG))
      )

      list(
        metrics   = do.call(rbind, rows_metrics),
        stability = stab_rows,
        selected  = if (length(rows_selected)) do.call(rbind, rows_selected) else NULL
      )
    },
    future.seed = TRUE
  )

  metrics_all   <- do.call(rbind, lapply(per_point, `[[`, "metrics"))
  stability_all <- do.call(rbind, lapply(per_point, `[[`, "stability"))
  selected_all  <- do.call(rbind, lapply(per_point, function(x) if (!is.null(x$selected)) x$selected else NULL))

  list(
    idx_points = idx_points,
    metrics    = metrics_all,
    stability  = stability_all,
    selected   = selected_all
  )
}

## 7) Experiment runner
run_realdata_experiment <- function(raw_df, target = "MMSCORE",
                                    test_frac = 0.3, K = 30,
                                    n_points = 20, seeds = 1:3,
                                    nsim_KernelSHAP = 300, n_perms_lime = 600,
                                    n_samples_bayes = 2000,
                                    stan_iter = 1000, stan_chains = 2,
                                    use_vi = FALSE,
                                    out_dir = "realdata_results") {
  target_resolved <- resolve_target_col(raw_df, target)                             
  message(sprintf("[run] Using target column: %s", target_resolved))       
  pre <- prep_adni_baseline(data = raw_df, target = target_resolved, task = "regression")  # prepare data

  set.seed(2048)                                                                    
  n <- nrow(pre$X); idx <- sample.int(n)                                          
  n_test <- max(1L, round(test_frac * n))                                      
  test_id  <- idx[seq_len(n_test)]                                             
  train_id <- setdiff(idx, test_id)                                             
  X_train <- pre$X[train_id, , drop = FALSE]; y_train <- pre$y[train_id]      
  X_test  <- pre$X[test_id,  , drop = FALSE]; y_test  <- pre$y[test_id]         
  is_const <- vapply(as.data.frame(X_train),
                   function(v) is.finite(sd(v, na.rm = TRUE)) && sd(v, na.rm = TRUE) == 0,
                   logical(1))
if (any(is_const)) {
  dropped_vars <- colnames(X_train)[is_const]
  message(sprintf("[pre] Dropping %d constant feature(s): %s",
                  length(dropped_vars), paste(dropped_vars, collapse = ", ")))
  X_train <- X_train[, !is_const, drop = FALSE]
  X_test  <- X_test[,  !is_const, drop = FALSE]
}

bad_col <- vapply(as.data.frame(X_train),
                  function(v) all(!is.finite(v)) || all(is.na(v)), logical(1))
if (any(bad_col)) {
  dropped2 <- colnames(X_train)[bad_col]
  message(sprintf("[pre] Dropping %d all-bad feature(s): %s",
                  length(dropped2), paste(dropped2, collapse = ", ")))
  X_train <- X_train[, !bad_col, drop = FALSE]
  X_test  <- X_test[,  !bad_col,  drop = FALSE]
}

feat_names <- make.names(colnames(X_train), unique = TRUE)   
  colnames(X_train) <- feat_names                    
  colnames(X_test)  <- feat_names                    

fit_spinn <- train_spinn(
  X_train, y_train,
  seed = 42L, epochs = 15L, batch_size = 64L,
  l1_input = 1e-3, verbose = 0L
)

spinn_scores <- spinn_feature_scores(fit_spinn)  

lin_res <- run_setting(
  X_train, y_train, X_test, y_test,
  model_type = "linear", K = K, seeds = seeds,
  n_points = n_points, nsim_KernelSHAP = nsim_KernelSHAP,
  n_perms_lime = n_perms_lime, n_samples_bayes = n_samples_bayes,
  stan_iter = stan_iter, stan_chains = stan_chains, use_vi = use_vi,
  spinn_scores = spinn_scores     
)

  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)    

    if (!is.null(lin_res)) {
    fwrite(lin_res$metrics,   file = file.path(out_dir, "metrics_linear.csv"))
    fwrite(lin_res$stability, file = file.path(out_dir, "stability_linear.csv"))
    if (!is.null(lin_res$selected))
      fwrite(lin_res$selected, file = file.path(out_dir, "selected_linear.csv")) 
  }

  combine_and_summarise <- function(res, tag) {
    if (is.null(res)) return(NULL)     
    m <- res$metrics %>%
      dplyr::group_by(model, method) %>%
      dplyr::summarise(
        infidelity_mean = mean(infidelity), infidelity_sd = sd(infidelity),       
        sparsity_mean   = mean(sparsity_topk_zeros), sparsity_sd = sd(sparsity_topk_zeros),
        runtime_mean    = mean(runtime_sec),        runtime_sd = sd(runtime_sec),
        .groups = "drop")
    s <- res$stability %>%
      dplyr::group_by(model, method) %>%
      dplyr::summarise(stability_mean = mean(stability_cos), stability_sd = sd(stability_cos),
                       .groups = "drop")
    out <- dplyr::left_join(m, s, by = c("model","method"))                        
    fwrite(out, file = file.path(out_dir, paste0("summary_", tag, ".csv")))       
    out                                                                       
  }

  sum_lin <- combine_and_summarise(lin_res, "linear")                      

list(
  linear = lin_res,
  summary_linear = sum_lin,
  paths = list(
    dir = out_dir,
    linear_metrics = file.path(out_dir, "metrics_linear.csv"),
    linear_summary = file.path(out_dir, "summary_linear.csv"),
    linear_selected = file.path(out_dir, "selected_linear.csv")
  )
)
}

## 9) MAIN: read data and run
raw <- readxl::read_excel("ADNI_with_MMSE.xlsx")                                   # load dataset
raw <- as.data.frame(raw)                                                          

out <- run_realdata_experiment(                
  raw_df = raw,        
  target = "MMSCORE",       
  test_frac = 0.3,                     
  K = 30,                
  n_points = 20,              
  seeds = 1:3,                                                                       # seeds for stability
  nsim_KernelSHAP = 300,                                                             # SHAP Monte Carlo samples
  n_perms_lime = 600,                                                                # LIME perturbation count
  n_samples_bayes = 2000,                                                            # BayesSLE perturbations
  stan_iter = 1000,                                                                  # BayesSLE Stan iterations
  stan_chains = 2,                                                                   # BayesSLE Stan chains
  use_vi = FALSE,                                                                    # VI off (HMC default)
  out_dir = "realdata_results"        
)

print(out$summary_linear)  

# Generate figures
make_real_linear_plots(
  out_dir         = out$paths$dir,
  formats         = c("pdf","png"),  
  weight_min_raw  = 0.10,  
  weight_min_norm = NULL,
  topN_per_method = 20,
  font_family     = NULL,         
  use_showtext    = FALSE,       
  width           = 8, height = 5, dpi = 300,
  title_infidelity = "Infidelity by method (linear model)",
  title_topvars    = "Top variables by method (linear model)"
)