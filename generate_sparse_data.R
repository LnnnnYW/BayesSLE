# generate_sparse_data.R
# Generate linear model data with sparse signal (flexible s, p, H2, rho)

generate_sparse_data <- function(n = 500, p = 1000, s = 10, H2 = 0.8, rho = 0.8, seed = 42) {
  set.seed(seed)
  
  # Construct correlation structure X (AR(1))
  X <- matrix(0, nrow = n, ncol = p)
  X[, 1] <- rnorm(n)
  for (j in 2:p) {
    X[, j] <- rho * X[, j - 1] + sqrt(1 - rho^2) * rnorm(n)
  }
  
  # Construct sparse beta
  active_idx <- sample(p, s)
  beta <- numeric(p)
  beta[active_idx] <- rnorm(s, mean = 1, sd = 0.2) * sample(c(-1, 1), s, TRUE)
  
  # Fitting y
  fx <- X %*% beta
  var_fx <- var(as.vector(fx))
  sigma2 <- (1 - H2) / H2 * var_fx
  y <- as.vector(fx) + rnorm(n, sd = sqrt(sigma2))
  
  colnames(X) <- paste0("V", 1:ncol(X))
  
  return(list(X = X, y = y, beta = beta, active_idx = active_idx))
}
