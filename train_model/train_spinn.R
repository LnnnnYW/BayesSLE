# code/models/train_spinn.R
train_spinn <- function(X, y,
                        lambda_path = 10^seq(-5, -1, length.out = 8),
                        hidden_units = 64,
                        learning_rate = 1e-3,
                        patience = 15,
                        epochs = 100,
                        batch_size = 32,
                        K = 30,
                        seed = 42,
                        save_model_path = NULL,
                        return_best_model = TRUE) {

  suppressPackageStartupMessages({
    library(keras3); library(tensorflow); library(reticulate)
  })
  set.seed(seed)

  n <- nrow(X); p <- ncol(X)
  idx <- sample.int(n)
  train_idx <- idx[1:floor(0.8 * n)]
  valid_idx <- idx[(floor(0.8 * n) + 1):n]
  X_train <- X[train_idx, ]; X_valid <- X[valid_idx, ]
  y_train <- y[train_idx]; y_valid <- y[valid_idx]

  X_train_scaled <- scale(X_train)
  center_attr <- attr(X_train_scaled, "scaled:center")
  scale_attr <- attr(X_train_scaled, "scaled:scale")
  X_valid_scaled <- sweep(sweep(X_valid, 2, center_attr, "-"), 2, scale_attr, "/")

  to_tf <- \(m) tensorflow::tf$cast(m, "float32")
  X_train_tf <- to_tf(X_train_scaled)
  X_valid_tf <- to_tf(X_valid_scaled)
  y_train_tf <- tensorflow::tf$reshape(to_tf(y_train), c(-1L, 1L))
  y_valid_tf <- tensorflow::tf$reshape(to_tf(y_valid), c(-1L, 1L))

  reticulate::py_run_string("
import tensorflow as tf
@tf.keras.utils.register_keras_serializable(package='Custom', name='GLcol')
class GLcol(tf.keras.regularizers.Regularizer):
    def __init__(self, lam):
        self.lam = float(lam)
    def __call__(self, w):
        row_norm = tf.sqrt(tf.reduce_sum(tf.square(w), axis=1)+1e-8)
        return self.lam * tf.reduce_sum(row_norm)
    def get_config(self):
        return {'lam': self.lam}
")
  GLcol <- reticulate::py$GLcol

  build_model <- function(lambda_gl) {
    inp <- layer_input(shape = p)
    h1  <- layer_dense(inp, hidden_units, activation = "tanh",
                       kernel_regularizer = GLcol(lambda_gl))
    out <- layer_dense(h1, 1)
    mdl <- keras_model(inp, out)
    mdl %>% compile(optimizer = optimizer_adam(learning_rate), loss = "mse")
    mdl
  }

  train_loss_vec <- numeric(length(lambda_path))
  valid_loss_vec <- numeric(length(lambda_path))
  models <- vector("list", length(lambda_path))

  for (i in seq_along(lambda_path)) {
    lambda_i <- lambda_path[i]
    model_i <- build_model(lambda_i)
    hist_i <- model_i %>% fit(
      X_train_tf, y_train_tf,
      validation_data = list(X_valid_tf, y_valid_tf),
      epochs = epochs,
      batch_size = batch_size,
      verbose = 0,
      callbacks = callback_early_stopping("val_loss", patience, restore_best_weights = TRUE)
    )
    train_loss_vec[i] <- min(hist_i$metrics$loss)
    valid_loss_vec[i] <- min(hist_i$metrics$val_loss)
    models[[i]] <- model_i
  }

  best_idx <- which.min(valid_loss_vec)
  best_model <- models[[best_idx]]

  if (!is.null(save_model_path)) best_model$save(save_model_path)

  if (return_best_model) best_model else
    list(models = models, lambda_path = lambda_path,
         valid_loss_vec = valid_loss_vec, best_idx = best_idx)
}
