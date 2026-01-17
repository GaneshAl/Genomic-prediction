predict_fold_RKHS <- function(y, D2, train_idx, test_idx,
                              eps = 1e-8) {

  ytr <- y[train_idx]
  mu  <- mean(ytr, na.rm = TRUE)
  yc  <- ytr - mu

  # train-only bandwidth (median heuristic on train block)
  D2_tr <- D2[train_idx, train_idx, drop = FALSE]
  off <- D2_tr[upper.tri(D2_tr)]
  h <- stats::median(off, na.rm = TRUE)
  if (!is.finite(h) || h <= 0) h <- mean(off[is.finite(off) & off > 0], na.rm = TRUE)
  if (!is.finite(h) || h <= 0) h <- 1.0

  K_tr <- exp(-D2_tr / (2 * h))
  K_te <- exp(-D2[test_idx, train_idx, drop = FALSE] / (2 * h))

  # ridge parameter (lambda) estimated from train only.
  # simple default: lambda = 1e-3 * mean(diag(K_tr)) to stabilize
  lambda <- 1e-3 * mean(diag(K_tr))
  if (!is.finite(lambda) || lambda <= 0) lambda <- 1e-3

  ntr <- length(train_idx)
  A <- K_tr + diag(lambda + eps, ntr)

  # solve for alpha
  alpha <- as.numeric(solve(A, yc))

  as.numeric(mu + K_te %*% alpha)
}

