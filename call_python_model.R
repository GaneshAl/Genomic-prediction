suppressPackageStartupMessages({
  library(data.table)
})

# Robust Python wrapper for fold-wise model calls.
# Allows optional extra args and environment variables (used for GAT ablations).
call_py_model <- function(script, train_df, test_df, out_pred,
                          python = "python",
                          tag = NULL,
                          extra_args = character(0),
                          env = character(0)) {
  dir.create("tmp", showWarnings = FALSE, recursive = TRUE)

  # unique, collision-free temp files
  if (is.null(tag) || !nzchar(tag)) {
    tag <- paste0("job_", as.integer(Sys.time()), "_", sprintf("%06d", sample.int(1e6, 1)))
  }
  train_path <- file.path("tmp", paste0("fold_train_", tag, ".csv"))
  test_path  <- file.path("tmp", paste0("fold_test_",  tag, ".csv"))

  fwrite(train_df, train_path)
  fwrite(test_df,  test_path)

  # Build command
  args <- c(script, train_path, test_path, out_pred, extra_args)

  # Run with optional env vars (NAME=VALUE strings)
  res <- tryCatch(
    system2(python,
            args = args,
            stdout = TRUE,
            stderr = TRUE,
            env = env),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    stop("Python model call failed (R error): ", conditionMessage(res))
  }
  if (!file.exists(out_pred)) {
    stop("Python model failed to write output: ", out_pred,
         "\nLast output:\n", paste(tail(res, 50), collapse = "\n"))
  }

  pred <- fread(out_pred)
  if (!all(c("ID", "yhat") %in% names(pred))) {
    stop("pred file must contain columns ID,yhat: ", out_pred)
  }
  pred
}

