explain_bayes_sle <- function(
  X, y, model, K = 50, seed = 123, spinn_model = NULL,
  n_samples = 2000, stan_iter = 1000, stan_chains = 2,
  use_vi = TRUE, active_idx = NULL
) {
  suppressPackageStartupMessages({
    if (!requireNamespace("glmnet", quietly = TRUE)) stop("Package 'glmnet' is required.")
    if (!requireNamespace("digest", quietly = TRUE)) stop("Package 'digest' is required.")
    # ranger is used as a black-box predictor variant; load if present
    if (requireNamespace("ranger", quietly = TRUE)) library(ranger)
    # keras/tensorflow/reticulate are optional; used only if SPINN is trained/read
    have_keras <- requireNamespace("keras3", quietly = TRUE) || requireNamespace("keras", quietly = TRUE)
    have_tf <- requireNamespace("tensorflow", quietly = TRUE)
    have_retic <- requireNamespace("reticulate", quietly = TRUE)
    # Stan backend: prefer cmdstanr; otherwise use rstan
    use_cmdstanr <- requireNamespace("cmdstanr", quietly = TRUE)
    if (!use_cmdstanr && !requireNamespace("rstan", quietly = TRUE))
      stop("Need either 'cmdstanr' or 'rstan' installed.")
  })

  ensure_feature_names <- function(M) {
    # Enforce canonical column names 'V1..Vp' to avoid mismatches at predict-time.
    M <- as.matrix(M)                                        
    colnames(M) <- paste0("V", seq_len(ncol(M)))             
    M                                                        # return matrix with names
  }
  predict_blackbox_numeric <- function(mdl, Xnew) {
    # Unify predictions across common model classes and return a numeric vector.
          if (inherits(mdl, "randomForest")) {
      yhat <- predict(mdl, newdata = as.data.frame(Xnew))    
    } else if ("ranger" %in% class(mdl)) {
      df <- as.data.frame(Xnew)                              
      yhat <- tryCatch(
        predict(mdl, data = df)$predictions,                 # exported predict
        error = function(e) ranger:::predict.ranger(mdl, data = df)$predictions 
      )
    } else if (inherits(mdl, "svm")) {
      yhat <- as.numeric(predict(mdl, newdata = Xnew))       
    } else if (inherits(mdl, c("cv.glmnet", "glmnet"))) {
      yhat <- as.numeric(predict(mdl, as.matrix(Xnew)))      
    } else if (inherits(mdl, "lin_interaction")) {
      yhat <- as.numeric(predict(mdl, as.matrix(Xnew)))      # custom linear-interaction model
    } else if (inherits(mdl, c("keras.src.models.sequential.Sequential",
                               "keras.src.models.functional.Functional"))) {
      yhat <- as.numeric(predict(mdl, as.matrix(Xnew)))      # keras model predictions
    } else if (is.function(mdl)) {
      yhat <- as.numeric(mdl(as.matrix(Xnew)))               
    } else {
      stop("Unsupported model type.")
    }
    if (is.matrix(yhat)) yhat <- yhat[, 1]                   # drop extra column if any
    as.numeric(yhat)                                         
  }
  cosine_distance_rows <- function(A, b, eps = 1e-12) {
    # Compute cosine distance between each row of A and vector b robustly.
    A <- as.matrix(A)                                        
    b <- as.numeric(b)                                       
    dots <- as.numeric(A %*% b)                              
    An <- sqrt(rowSums(A^2) + eps)                           
    bn <- sqrt(sum(b^2) + eps)                               
    cs <- pmin(pmax(dots / (An * bn), -1.0), 1.0)            
    1 - cs                                                   # return cosine distance
  }

  stan_fit_vi_then_hmc <- function(sm, data_list, seed, use_vi, use_cmdstanr,
                                   stan_iter, stan_chains) {
    # Try VI (fullrank then meanfield); if it fails, fall back to HMC sampling.
    get_beta_mat <- function(fit_obj) {
      # Extract draws matrix for 'beta' in a backend-agnostic way.
      bm <- tryCatch(fit_obj$draws("beta"), error = function(e) NULL)     
      if (is.null(bm)) bm <- tryCatch(as.matrix(fit_obj, pars = "beta"),  
                                      error = function(e) NULL)
      if (is.null(bm)) stop("Fitting failed. Unable to retrieve the draws.")
      if (!is.matrix(bm)) bm <- as.matrix(bm)                          
      # Subset beta columns if needed (cmdstanr may return a draws_df)
      if (!is.null(colnames(bm))) {
        idx <- grep("^beta\\[", colnames(bm))
        if (length(idx)) bm <- bm[, idx, drop = FALSE]
      }
      bm
    }

    if (use_vi) {
      # attempt fullrank VI with conservative settings
      ok <- FALSE
      beta_mat <- NULL
      if (use_cmdstanr) {
        try({
          fit <- sm$variational(data = data_list, seed = seed, iter = 10000,
                                algorithm = "fullrank", eta = 0.02, grad_samples = 8,
                                elbo_samples = 20, tol_rel_obj = 0.001)
          beta_mat <- get_beta_mat(fit)
          if (all(is.finite(beta_mat))) ok <- TRUE
        }, silent = TRUE)
        if (!ok) {
          # fallback to meanfield VI
          try({
            fit <- sm$variational(data = data_list, seed = seed, iter = 8000,
                                  algorithm = "meanfield", eta = 0.02, grad_samples = 8,
                                  elbo_samples = 20, tol_rel_obj = 0.001)
            beta_mat <- get_beta_mat(fit)
            if (all(is.finite(beta_mat))) ok <- TRUE
          }, silent = TRUE)
        }
      } else {
        # rstan variational (single algorithm; rstan does not expose fullrank/meanfield option)
        try({
          fit <- rstan::vb(sm, data = data_list, seed = seed, iter = 8000, eta = 0.02)
          beta_mat <- get_beta_mat(fit)
          if (all(is.finite(beta_mat))) ok <- TRUE
        }, silent = TRUE)
      }
      if (ok) return(beta_mat)                               # successful VI draws
      message("[BayesSLE] VI failed or unstable; falling back to HMC.")
    }

    # HMC fallback
    if (use_cmdstanr) {
      fit <- sm$sample(data = data_list, seed = seed, chains = stan_chains,
                       iter_warmup = ceiling(stan_iter / 2),
                       iter_sampling = floor(stan_iter / 2),
                       refresh = 0)
    } else {
      fit <- rstan::sampling(sm, data = data_list, seed = seed,
                             iter = stan_iter, warmup = floor(stan_iter / 2),
                             chains = stan_chains, refresh = 0)
    }
    get_beta_mat(fit)                                        # return draws matrix
  }

  # inputs & ground truth
  if (is.null(active_idx)) active_idx <- attr(X, "active_idx")  
  if (is.null(active_idx)) active_idx <- integer(0)             
  X <- ensure_feature_names(X)                                  
  p <- ncol(X)    

  # SPINN score (structure-aware keep probability)
  s_norm <- rep(0, p)                                           # default: uninformative zeros
  if (is.null(spinn_model)) {
    # Try to train SPINN only if user provides a train_spinn() and keras/tf is present.
    if (exists("train_spinn", mode = "function")) {
      set.seed(seed)                                            # seed for SPINN training reproducibility
      spinn_model <- tryCatch(train_spinn(X, y, K = K, seed = seed), error = function(e) NULL)
    }
  }
  if (!is.null(spinn_model)) {
    # Extract first Dense layer kernel weights and compute row (feature) L2 norms.
    layers_list <- tryCatch(spinn_model$layers, error = function(e) NULL)
    if (is.function(layers_list)) layers_list <- tryCatch(layers_list(), error = function(e) NULL)
    # If reticulate is available, coerce python list to R
    if (have_retic) layers_list <- tryCatch(reticulate::py_to_r(layers_list), error = function(e) layers_list)
    # Find the first Dense layer and read its kernel weights
    dense_idx <- tryCatch(which(vapply(
      layers_list, function(l) grepl("Dense", toString(class(l)), TRUE), logical(1))), error = function(e) integer(0))
    if (length(dense_idx) > 0) {
      W1 <- tryCatch(layers_list[[dense_idx[1]]]$get_weights()[[1]], error = function(e) NULL)
      if (is.null(W1)) W1 <- tryCatch(layers_list[[dense_idx[1]]]$weights[[1]]$numpy(), error = function(e) NULL)
      if (is.matrix(W1) || is.array(W1)) {
        spinn_score <- sqrt(rowSums(matrix(W1, nrow = p)[, , drop = FALSE]^2))  # row L2 over units
        # Normalize to [0,1] robustly
        rng <- max(spinn_score) - min(spinn_score)
        if (is.finite(rng) && rng > 0) {
          s_norm <- (spinn_score - min(spinn_score)) / (rng + 1e-12)
        } else {
          s_norm <- rep(0, p)
        }
      }
    }
  }
  # Convert scores to keep probabilities: alpha + (1-alpha) * s_j
  alpha_floor <- 0.1                                          
  keep_prob <- alpha_floor + (1 - alpha_floor) * s_norm       # keep probability per feature in [0.1, 1]

  # choose an instance to explain
set.seed(seed)
x0_idx <- sample.int(nrow(X), 1L)
x0_vec <- as.numeric(X[x0_idx, , drop = TRUE])
baseline <- colMeans(X)

# structured masks: Z=1 means "set to baseline", Z=0 means "keep x0"
# Convert SPINN scores (s_norm in [0,1]) to masking probabilities p_mask in [rho_min, rho_max].
# Higher s_norm => less likely to be masked.
rho_min <- 0.1
rho_max <- 0.9
p_mask  <- rho_min + (rho_max - rho_min) * (1 - s_norm)

Z <- matrix(
  rbinom(n_samples * p, size = 1L, prob = rep(p_mask, each = n_samples)),
  nrow = n_samples, ncol = p
)
# Add anchor masks: all-zeros => x0; all-ones => baseline
Z <- unique(rbind(Z, rep(0L, p), rep(1L, p)))

# Build perturbed inputs: X' = (1-Z)*x0 + Z*baseline
X_pert <- sweep(1L - Z, 2L, x0_vec, "*") + sweep(Z, 2L, baseline, "*")
colnames(X_pert) <- colnames(X)

# Design matrix in masking-amplitude space: DZ = Z * (baseline - x0)
Delta <- baseline - x0_vec
DZ    <- sweep(Z, 2L, Delta, "*")

# black-box predictions and local response dy = f(X') - f(x0)
fx     <- as.numeric(predict_blackbox_numeric(model, matrix(x0_vec, 1L)))
y_pert <- as.numeric(predict_blackbox_numeric(model, X_pert))

dy <- y_pert - fx

# Small jitter only if degenerate
if (abs(stats::sd(dy)) < 1e-12) {
  set.seed(seed + 999)
  dy <- dy + rnorm(length(dy), 0, 1e-6)
}

# kernel weights: Gaussian over cosine distance on X'
# Cosine distance: d in [0,2], bandwidth = median positive distance 
d <- 1 - as.numeric((X_pert %*% x0_vec) /
      (sqrt(rowSums(X_pert^2)) * sqrt(sum(x0_vec^2)) + 1e-12))
lambda <- stats::median(d[d > 0])
if (!is.finite(lambda) || lambda <= 0) lambda <- 1.0
w <- exp(-(d^2) / (lambda^2))
w <- w / mean(w)

# weighted Elastic Net screening to pick Top-K mask columns
en_cv <- glmnet::cv.glmnet(
  x = DZ, y = dy, weights = w,
  alpha = 0.5, intercept = TRUE, standardize = TRUE, nfolds = 5, type.measure = "mse"
)
imp <- abs(as.numeric(stats::coef(en_cv, s = "lambda.min"))[-1])
imp[is.na(imp)] <- 0
K_scr <- min(K, length(imp))
topk_vars <- head(order(imp, decreasing = TRUE), K_scr)

# Drop degenerate columns under weights
wbar <- sum(w)
col_var_w <- vapply(topk_vars, function(j) {
  zj <- DZ[, j]; m <- sum(w * zj) / wbar
  sum(w * (zj - m)^2) / wbar
}, numeric(1))
keep_idx <- which(col_var_w > 1e-10)
topk_vars <- topk_vars[keep_idx]
if (!length(topk_vars)) stop("All selected columns are degenerate; increase n_samples.")

# weighted standardization for Stan on DZ[:, topk_vars]
Zsel   <- DZ[, topk_vars, drop = FALSE]
z_mean <- as.numeric(colSums(w * Zsel) / wbar)
Zc     <- sweep(Zsel, 2L, z_mean, "-")
z_sd   <- sqrt(pmax(colSums(w * (Zc^2)) / wbar, 1e-12))
Zs     <- sweep(Zc, 2L, z_sd, "/")
y_mean <- sum(w * dy) / wbar
ys     <- dy - y_mean

  # Stan model
  stan_code <- "
  data {
    int<lower=1> N;                 // number of samples
    int<lower=1> P;                 // number of predictors
    matrix[N, P] Z;                 // standardized design matrix
    vector[N] y;                    // centered response
    vector<lower=0>[N] w;           // kernel weights (clipped and mean-normalized)
    real<lower=0> lambda;           // Laplace prior rate
  }
  parameters {
    real alpha;                     // intercept on centered scale
    vector[P] beta;                 // coefficients on standardized scale
    real<lower=0> sigma;            // noise scale
  }
  model {
    alpha ~ normal(0, 5);           // weak prior on intercept
    beta  ~ double_exponential(0, 1 / lambda); // Laplace(scale = 1/lambda)
    sigma ~ student_t(3, 0, 2.5);   // half-Student-t prior (implicit truncation)
    for (n in 1:N) {
      y[n] ~ normal(alpha + Z[n] * beta, sigma / sqrt(fmax(w[n], 1e-12))); // weighted likelihood
    }
  }"

  # compile
  use_cmdstanr <- requireNamespace("cmdstanr", quietly = TRUE) 
  stan_hash  <- digest::digest(stan_code)                      
  model_name <- paste0("bayes_lasso_", stan_hash)             
  if (!exists(model_name, .GlobalEnv)) {
    assign(model_name,
      if (use_cmdstanr) {
        cmdstanr::cmdstan_model(cmdstanr::write_stan_file(stan_code))     
      } else {
        rstan::stan_model(model_code = stan_code)     
      },
      envir = .GlobalEnv
    )
  }
  sm <- get(model_name, .GlobalEnv)  

    # fit across lambda grid with VI->HMC fallback; choose best by weighted R^2
  lambda_grid <- c(0.001, 0.005, 0.01, 0.05, 0.1)
  best_obj   <- NULL
  best_score <- -Inf

  for (lam in lambda_grid) {
    stan_data <- list(
      N = nrow(Zs), P = ncol(Zs),
      Z = Zs, y = as.numeric(ys),
      w = as.numeric(w), 
      lambda = lam
    )

    # Fit and get posterior draws of beta on standardized scale
    beta_mat <- stan_fit_vi_then_hmc(
      sm, stan_data, seed = seed, use_vi = use_vi,
      use_cmdstanr = use_cmdstanr, stan_iter = stan_iter, stan_chains = stan_chains
    )

    # Summaries on standardized scale
    b_mean_std <- colMeans(beta_mat)
    b_sd_std   <- apply(beta_mat, 2, sd)

    # Map back to unstandardized DZ columns
    b_mean <- b_mean_std / z_sd
    b_sd   <- b_sd_std  / z_sd

    # Fill to full length
    phi_full_mean <- phi_full_sd <- numeric(p)
    phi_full_mean[topk_vars] <- b_mean
    phi_full_sd[topk_vars]   <- b_sd

    # Z-scale contributions
    contrib_z <- phi_full_mean * Delta

    # Scoring: weighted R^2 on the perturbation set
    mu <- as.numeric(Zsel %*% b_mean)
    # exact intercept back-transform: y_mean - sum(b_mean * z_mean)
    mu <- mu + (y_mean - sum(b_mean * z_mean))
    rss <- sum(w * (as.numeric(dy) - mu)^2)
    tss <- sum(w * (as.numeric(dy) - mean(dy))^2)
    score <- 1 - rss / max(tss, 1e-12)

    if (is.finite(score) && score > best_score) {
      best_score <- score
      best_obj <- list(
        phi_full_mean = phi_full_mean,
        phi_full_sd   = phi_full_sd,
        contrib_z     = contrib_z,
        topk_vars     = topk_vars,
        lambda        = lam
      )
    }
  }  

  if (is.null(best_obj)) {
    stop("No valid model selected; check sampling and lambda grid.")
  }

  # Build posterior data frame from best object
  posterior <- data.frame(
    variable = colnames(X),
    mean     = as.numeric(best_obj$contrib_z),
    sd       = as.numeric(abs(Delta) * best_obj$phi_full_sd)
  )

  # Final return
  return(list(
    topk_vars   = best_obj$topk_vars,
    posterior   = posterior,
    best_lambda = best_obj$lambda,
    spinn_score = s_norm,
    x0_index    = x0_idx,
    attr_x      = as.numeric(best_obj$phi_full_mean),
    attr_z      = as.numeric(best_obj$contrib_z)
  ))
} 
