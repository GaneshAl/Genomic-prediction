suppressPackageStartupMessages({
  library(BGLR)
})

predict_fold_BayesB <- function(y, M, train_idx, test_idx,
                                nIter = 12000, burnIn = 4000, thin = 5) {
  y_cv <- y
  y_cv[test_idx] <- NA
  
  fit <- BGLR(
    y = y_cv,
    ETA = list(list(X = M, model = "BayesB")),
    nIter = nIter, burnIn = burnIn, thin = thin,
    verbose = FALSE
  )
  as.numeric(fit$yHat[test_idx])
}
