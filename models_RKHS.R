suppressPackageStartupMessages({
  library(BGLR)
})

predict_fold_RKHS <- function(y, K, train_idx, test_idx,
                              nIter = 6000, burnIn = 2000, thin = 5) {
  y_cv <- y
  y_cv[test_idx] <- NA
  
  fit <- BGLR(
    y = y_cv,
    ETA = list(list(K = K, model = "RKHS")),
    nIter = nIter, burnIn = burnIn, thin = thin,
    verbose = FALSE
  )
  as.numeric(fit$yHat[test_idx])
}
