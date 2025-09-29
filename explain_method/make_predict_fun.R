make_predict_fun <- function(model) {
  force(model)

  # glmnet / cv.glmnet / lasso_glmnet
  } else if (inherits(model, "cv.glmnet")) {
    function(newdata) {
      as.numeric(glmnet::predict.glmnet(model$glmnet.fit, newx = as.matrix(newdata), s = model$lambda.min))
    }
  } else if (inherits(model, "lasso_glmnet")) {
    function(newdata) {
      s_use <- attr(model, "best_lambda"); if (is.null(s_use)) s_use <- "lambda.min"
      as.numeric(glmnet::predict.glmnet(model, newx = as.matrix(newdata), s = s_use))
    }
  } else if (inherits(model, "glmnet")) {
    function(newdata) {
      s_use <- attr(model, "best_lambda"); if (is.null(s_use)) s_use <- "lambda.min"
      as.numeric(glmnet::predict.glmnet(model, newx = as.matrix(newdata), s = s_use))
    }

  } else if (inherits(model, "lin_interaction")) {
    function(newdata) as.numeric(predict(model, newx = as.matrix(newdata)))

  # keras models
  } else if (inherits(model, "keras.src.models.sequential.Sequential") ||
             inherits(model, "keras.src.models.functional.Functional") ||
             inherits(model, "keras.Model") ||
             inherits(model, "tensorflow.python.keras.engine.training.Model")) {
    function(newdata) as.numeric(model %>% predict(as.matrix(newdata)))

  } else {
    function(newdata) as.numeric(predict(model, newdata = as.data.frame(newdata)))
  }
}
