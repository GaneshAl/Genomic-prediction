suppressPackageStartupMessages({
  library(rrBLUP)
})

predict_fold_rrBLUP <- function(y, M, train_idx, test_idx) {
  Ztr <- M[train_idx, , drop = FALSE]
  ms <- mixed.solve(y = y[train_idx], Z = Ztr)
  u  <- as.numeric(ms$u)
  b0 <- as.numeric(ms$beta)
  
  Zte <- M[test_idx, , drop = FALSE]
  as.numeric(Zte %*% u) + b0
}
