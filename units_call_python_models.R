suppressPackageStartupMessages({
  library(data.table)
})

call_py_model <- function(script, train_df, test_df, out_pred,
                          python = "python") {
  dir.create("tmp", showWarnings = FALSE, recursive = TRUE)
  
  train_path <- file.path("tmp", "fold_train.csv")
  test_path  <- file.path("tmp", "fold_test.csv")
  
  fwrite(train_df, train_path)
  fwrite(test_df,  test_path)
  
  cmd <- sprintf('%s "%s" "%s" "%s" "%s"',
                 python, script, train_path, test_path, out_pred)
  status <- system(cmd, intern = FALSE, ignore.stdout = TRUE, ignore.stderr = FALSE)
  if (!identical(status, 0L)) stop("Python model failed: ", script)
  
  pred <- fread(out_pred)
  if (!all(c("ID","yhat") %in% names(pred))) {
    stop("pred file must contain columns ID,yhat: ", out_pred)
  }
  pred
}
