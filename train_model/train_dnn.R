# train_dnn.R

train_dnn <- function(X, y, save_path = NULL, seed = 42) {
  suppressPackageStartupMessages({
    library(keras3); library(tensorflow); library(reticulate)
  })
  set.seed(seed)
  
  # Divide the training & validation set
  n <- nrow(X)
  idx <- sample.int(n)
  train_idx <- idx[1:floor(0.8 * n)]
  valid_idx <- idx[(floor(0.8 * n) + 1):n]
  
  X_train <- X[train_idx, ]
  X_valid <- X[valid_idx, ]
  y_train <- y[train_idx]
  y_valid <- y[valid_idx]
  
  # Scaling
  X_train_scaled <- scale(X_train)
  center_attr <- attr(X_train_scaled, "scaled:center")
  scale_attr  <- attr(X_train_scaled, "scaled:scale")
  X_valid_scaled <- sweep(sweep(X_valid, 2, center_attr, "-"), 2, scale_attr, "/")
  
  to_tf <- \(m) tensorflow::tf$cast(m, "float32")
  X_train_tf <- to_tf(X_train_scaled)
  X_valid_tf <- to_tf(X_valid_scaled)
  y_train_tf <- tensorflow::tf$reshape(to_tf(y_train), c(-1L, 1L))
  y_valid_tf <- tensorflow::tf$reshape(to_tf(y_valid), c(-1L, 1L))
  
  # Hyperparameters
  learning_rate <- 1e-3
  patience <- 15
  epochs <- 100
  batch_size <- 32
  
  # Modified network architecture
  input <- layer_input(shape = ncol(X))
  output <- input %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  
  model <- keras_model(input, output)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate),
    loss = "mse"
  )
  
  # Training
  model %>% fit(
    X_train_tf, y_train_tf,
    validation_data = list(X_valid_tf, y_valid_tf),
    epochs = epochs,
    batch_size = batch_size,
    verbose = 0,
    callbacks = callback_early_stopping("val_loss", patience, restore_best_weights = TRUE)
  )
  
  if (!is.null(save_path)) {
    model$save(save_path)
  }
  
  return(model)
}
