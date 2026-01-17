suppressPackageStartupMessages({
  library(BGLR)
})

predict_fold_BayesB <- function(y, M, train_idx, test_idx,
                                nIter = 12000, burnIn = 4000, thin = 5) {

  ytr <- y[train_idx]
  Xtr <- M[train_idx, , drop = FALSE]
  Xte <- M[test_idx,  , drop = FALSE]

  fit <- BGLR(
    y = ytr,
    ETA = list(list(X = Xtr, model = "BayesB")),
    nIter = nIter, burnIn = burnIn, thin = thin,
    verbose = FALSE
  )

  # BGLR usually stores marker effects here for marker models
  b <- fit$ETA[[1]]$b
  if (is.null(b)) {
    stop("BayesB strict OOF: could not find marker effects fit$ETA[[1]]$b in BGLR output.")
  }

  mu <- as.numeric(fit$mu)
  as.numeric(mu + Xte %*% b)
}

