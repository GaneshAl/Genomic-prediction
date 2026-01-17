suppressPackageStartupMessages({
  library(data.table)
})

call_py_model <- function(script, train_df, test_df, out_pred,
                          python = "python",
                          tag = NULL) {
  dir.create("tmp", showWarnings = FALSE, recursive = TRUE)

  # unique, collision-free temp files
  if (is.null(tag) || !nzchar(tag)) {
    tag <- paste0("job_", as.integer(Sys.time()), "_", sprintf("%06d", sample.int(1e6, 1)))
  }
  train_path <- file.path("tmp", paste0("fold_train_", tag, ".csv"))
  test_path  <- file.path("tmp", paste0("fold_test_",  tag, ".csv"))

  fwrite(train_df, train_path)
  fwrite(test_df,  test_path)

  # robust invocation
  # args: script train test out_pred
  res <- tryCatch(
    system2(python, args = c(script, train_path, test_path, out_pred),
            stdout = TRUE, stderr = TRUE),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    stop("Python model call failed (R error): ", conditionMessage(res))
  }
  # system2 doesn't directly give exit status when stdout/stderr captured like this,
  # so we do a simple existence check:
  if (!file.exists(out_pred)) {
    stop("Python model failed to write output: ", out_pred, "\nLast output:\n",
         paste(tail(res, 50), collapse = "\n"))
  }

  pred <- fread(out_pred)
  if (!all(c("ID", "yhat") %in% names(pred))) {
    stop("pred file must contain columns ID,yhat: ", out_pred)
  }
  pred
}

