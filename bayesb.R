suppressPackageStartupMessages({
  library(BGLR)
})

predict_fold_BayesB <- function(y, M, train_idx, test_idx,
                                # Reduce the default number of iterations and burn-in to
                                # accelerate BayesB training. Fewer iterations yield lower
                                # runtime while still producing reasonable estimates. BGLR's
                                # sampler parameters (nIter, burnIn, thin) control the number
                                # of MCMC iterations?17366928874999†L440-L442?.
                                nIter = 4000, burnIn = 1000, thin = 5) {

  ytr <- y[train_idx]
  Xtr <- M[train_idx, , drop = FALSE]
  Xte <- M[test_idx,  , drop = FALSE]

  fit <- BGLR(
    y = ytr,
    ETA = list(list(X = Xtr, model = "BayesB")),
    # Pass through the reduced iteration parameters defined above. Shorter
    # chains reduce runtime without changing the BayesB methodology.
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

