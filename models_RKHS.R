# RKHS (Gaussian kernel ridge regression) for genomic prediction

predict_fold_RKHS <- function(y, M, train_idx, test_idx,
                              eps = 1e-8) {

  # y: length-n numeric
  # M: n x p numeric matrix (no NA), already foldwise preprocessed
  ytr <- y[train_idx]
  mu  <- mean(ytr, na.rm = TRUE)
  yc  <- ytr - mu

  Xtr <- M[train_idx, , drop = FALSE]
  Xte <- M[test_idx,  , drop = FALSE]

  # --- Efficient squared Euclidean distances ---
  # D2(a,b) = ||a||^2 + ||b||^2 - 2 a.b
  tr_sq <- rowSums(Xtr * Xtr)
  te_sq <- rowSums(Xte * Xte)
  G_tr  <- tcrossprod(Xtr)                      # ntr x ntr
  D2_tr <- outer(tr_sq, tr_sq, "+") - 2 * G_tr
  D2_tr[D2_tr < 0] <- 0

  G_te  <- Xte %*% t(Xtr)                       # nte x ntr
  D2_te <- outer(te_sq, tr_sq, "+") - 2 * G_te
  D2_te[D2_te < 0] <- 0

  # --- Train-only bandwidth (median heuristic on train block) ---
  off <- D2_tr[upper.tri(D2_tr)]
  h <- stats::median(off, na.rm = TRUE)
  if (!is.finite(h) || h <= 0) {
    h <- mean(off[is.finite(off) & off > 0], na.rm = TRUE)
  }
  if (!is.finite(h) || h <= 0) h <- 1.0

  # Gaussian kernel
  K_tr <- exp(-D2_tr / (2 * h))
  K_te <- exp(-D2_te / (2 * h))

  # Ridge parameter (lambda) train-only; simple stabilization
  lambda <- 1e-3 * mean(diag(K_tr))
  if (!is.finite(lambda) || lambda <= 0) lambda <- 1e-3

  ntr <- nrow(K_tr)
  A <- K_tr + diag(lambda + eps, ntr)

  alpha <- as.numeric(solve(A, yc))

  as.numeric(mu + K_te %*% alpha)
}

