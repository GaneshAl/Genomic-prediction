#!/usr/bin/env Rscript
# ==============================================================
# Fully Nested CV Genomic Prediction Pipeline
# rrBLUP + BayesB + RKHS + RF + SVR + GAT
#
# Outer CV: unbiased performance estimate
# Inner CV: generate OOF base predictions on outer-train set
#           + fit ridge stacker (with cv.glmnet lambda selection)
#
# Outputs:
# - metrics_overall_<trait>.tsv  (outer-CV aggregated)
# - metrics_by_outer_<trait>.tsv (per outer fold)
# - stacking_weights_<trait>.tsv (per outer fold weights)
# - runtime_by_model_<trait>.tsv
# ==============================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript run_driver_nested.R <phenotype_file>\n")
pheno_file <- args[1]

suppressPackageStartupMessages({
  library(data.table)
  library(rrBLUP)
  library(BGLR)
  library(glmnet)
})

# ---------------- USER SETTINGS ----------------
TRAITS <- c("BC", "FF", "FW", "TSS", "TC")
ID_COL <- "ID"

K_OUTER <- 5
K_INNER <- 5
N_REPS  <- 10
set.seed(123)

# BayesB iterations
BAYESB_nIter  <- 12000
BAYESB_burnIn <- 4000
BAYESB_thin   <- 5

# Python scripts
PYTHON <- "python"
RF_SCRIPT  <- "py_models/rf.py"
SVR_SCRIPT <- "py_models/svr.py"
GAT_SCRIPT <- "py_models/gat.py"

dir.create("results_nested", showWarnings = FALSE, recursive = TRUE)
dir.create("tmp", showWarnings = FALSE, recursive = TRUE)

# ---------------- SOURCE MODEL HELPERS ----------------
source("models/rrblup.R")    # predict_fold_rrBLUP(...)
source("models/bayesb.R")    # predict_fold_BayesB(...)
source("models/rkhs.R")      # predict_fold_RKHS(...)
source("utils/call_python_model.R")  # call_py_model(...)

# ---------------- HELPERS ----------------
safe_cor <- function(a, b) {
  ok <- complete.cases(a, b)
  if (sum(ok) < 2) return(NA_real_)
  if (sd(b[ok]) == 0) return(NA_real_)
  cor(a[ok], b[ok])
}

metrics_vec <- function(y_true, y_pred) {
  ok <- complete.cases(y_true, y_pred)
  if (sum(ok) < 2) return(list(r=NA_real_, spearman=NA_real_, rmse=NA_real_, mae=NA_real_))
  e <- y_true[ok] - y_pred[ok]
  list(
    r = safe_cor(y_true[ok], y_pred[ok]),
    spearman = suppressWarnings(cor(y_true[ok], y_pred[ok], method="spearman")),
    rmse = sqrt(mean(e^2)),
    mae  = mean(abs(e))
  )
}

fit_stack_ridge <- function(X, y) {
  # Fit ridge with internal CV on the *meta-training set only*
  # Store scaling so predict uses identical transform.
  Xs <- scale(X)
  ctr <- attr(Xs, "scaled:center")
  scl <- attr(Xs, "scaled:scale")

  cv <- cv.glmnet(x = Xs, y = y, alpha = 0)

  scale_with <- function(A) {
    sweep(sweep(A, 2, ctr, "-"), 2, scl, "/")
  }

  list(
    cv = cv,
    lambda = cv$lambda.min,
    coef = as.matrix(coef(cv, s="lambda.min")),
    predict = function(newX) as.numeric(predict(cv, scale_with(newX), s="lambda.min")),
    scale_center = ctr,
    scale_scale  = scl
  )
}

# Create fold labels for a given index vector
make_folds <- function(idx, K) {
  # returns integer labels 1..K for each element of idx
  sample(rep(seq_len(K), length.out = length(idx)))
}

# Build train/test data.frames for python models from genotype matrix M and phenotype y
make_py_train_test <- function(ids, M, y, train_idx, test_idx) {
  train_df <- data.table(ID = ids[train_idx], y = y[train_idx])
  test_df  <- data.table(ID = ids[test_idx])

  Xtr <- as.data.table(M[train_idx, , drop=FALSE])
  Xte <- as.data.table(M[test_idx,  , drop=FALSE])

  train_df <- cbind(train_df, Xtr)
  test_df  <- cbind(test_df,  Xte)

  list(train_df=train_df, test_df=test_df)
}

# ---------------- READ GENOTYPES ----------------
cat("Reading genotypes.\n")
geno <- fread("GP_MARKERS.raw")
ids <- geno$IID

M <- as.matrix(geno[, -(1:6), with = FALSE])
rownames(M) <- ids
colnames(M) <- colnames(geno)[-(1:6)]
rm(geno); gc()

# Remove zero-variance SNPs
v <- apply(M, 2, var, na.rm = TRUE)
M <- M[, v > 0, drop = FALSE]

# Mean impute + center (global X-only preprocessing; strict alternative is foldwise)
cm <- colMeans(M, na.rm = TRUE)
for (j in seq_len(ncol(M))) {
  nas <- is.na(M[, j])
  if (any(nas)) M[nas, j] <- cm[j]
}
M <- scale(M, center = TRUE, scale = FALSE)
M <- as.matrix(M)

# ---------------- READ PHENOTYPES ----------------
pheno <- fread(pheno_file)
if (!(ID_COL %in% names(pheno))) stop("Phenotype file must have ID column: ", ID_COL)

# align phenotypes to genotype IDs
setkeyv(pheno, ID_COL)
pheno <- pheno[J(ids), nomatch=0]
if (nrow(pheno) == 0) stop("No overlapping IDs between genotype and phenotype file.")
pheno_ids <- pheno[[ID_COL]]

# reorder genotype matrix to match phenotype ordering
ord <- match(pheno_ids, ids)
ids <- ids[ord]
M <- M[ord, , drop=FALSE]

# ---------------- MAIN LOOP ----------------
model_names <- c("rrBLUP", "BayesB", "RKHS", "RF", "SVR", "GAT")

for (trait in TRAITS) {
  cat("\n=== Trait:", trait, "===\n")
  if (!(trait %in% names(pheno))) {
    cat("Trait not found in phenotype file:", trait, " -- skipping\n")
    next
  }

  y <- as.numeric(pheno[[trait]])
  n <- length(y)

  outdir <- file.path("results_nested", trait)
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir, "metrics"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir, "stacking"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir, "runtime"), showWarnings = FALSE, recursive = TRUE)

  metrics_outer <- data.table(
    Trait=character(), Rep=integer(), OuterFold=integer(), Model=character(),
    r=numeric(), spearman=numeric(), rmse=numeric(), mae=numeric()
  )
  weight_log <- data.table(
    Trait=character(), Rep=integer(), OuterFold=integer(),
    Term=character(), Coef=numeric(), Lambda=numeric()
  )
  runtime_log <- data.table(Trait=character(), Rep=integer(), OuterFold=integer(),
                            Phase=character(), Model=character(), Seconds=numeric())

  # ---------- Repeated OUTER CV ----------
  idx_nonNA <- which(!is.na(y))
  for (rep in seq_len(N_REPS)) {
    folds_outer <- make_folds(idx_nonNA, K_OUTER)

    for (k_out in seq_len(K_OUTER)) {
      t_outer0 <- proc.time()[3]

      test_idx_outer  <- idx_nonNA[folds_outer == k_out]
      train_idx_outer <- setdiff(idx_nonNA, test_idx_outer)

      # ---------- INNER CV on OUTER-TRAIN to create meta-training OOF ----------
      t_inner0 <- proc.time()[3]

      folds_inner <- make_folds(train_idx_outer, K_INNER)

      # inner OOF preds for meta-training (rows=train_idx_outer only)
      oof_inner <- matrix(NA_real_, nrow=n, ncol=length(model_names),
                          dimnames=list(ids, model_names))

      for (k_in in seq_len(K_INNER)) {
        val_idx_inner <- train_idx_outer[folds_inner == k_in]
        fit_idx_inner <- setdiff(train_idx_outer, val_idx_inner)

        # rrBLUP
        t0 <- proc.time()[3]
        oof_inner[val_idx_inner, "rrBLUP"] <- predict_fold_rrBLUP(y, M, fit_idx_inner, val_idx_inner)
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="rrBLUP", Seconds=proc.time()[3]-t0))

        # BayesB
        t0 <- proc.time()[3]
        oof_inner[val_idx_inner, "BayesB"] <- predict_fold_BayesB(y, M, fit_idx_inner, val_idx_inner,
                                                                 nIter=BAYESB_nIter, burnIn=BAYESB_burnIn, thin=BAYESB_thin)
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="BayesB", Seconds=proc.time()[3]-t0))

        # RKHS
        t0 <- proc.time()[3]
        oof_inner[val_idx_inner, "RKHS"] <- predict_fold_RKHS(y, M, fit_idx_inner, val_idx_inner)
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="RKHS", Seconds=proc.time()[3]-t0))

        # Python models: RF / SVR / GAT
        # Build fold-specific train/test dfs
        py <- make_py_train_test(ids, M, y, fit_idx_inner, val_idx_inner)

        # RF
        t0 <- proc.time()[3]
        rf_out <- file.path("tmp", sprintf("pred_RF_%s_rep%02d_out%02d_in%02d.csv", trait, rep, k_out, k_in))
        pred_rf <- call_py_model(RF_SCRIPT, py$train_df, py$test_df, rf_out, python=PYTHON,
                                tag=sprintf("RF_%s_rep%02d_out%02d_in%02d", trait, rep, k_out, k_in))
        oof_inner[val_idx_inner, "RF"] <- pred_rf$yhat[match(ids[val_idx_inner], pred_rf$ID)]
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="RF", Seconds=proc.time()[3]-t0))

        # SVR
        t0 <- proc.time()[3]
        svr_out <- file.path("tmp", sprintf("pred_SVR_%s_rep%02d_out%02d_in%02d.csv", trait, rep, k_out, k_in))
        pred_svr <- call_py_model(SVR_SCRIPT, py$train_df, py$test_df, svr_out, python=PYTHON,
                                 tag=sprintf("SVR_%s_rep%02d_out%02d_in%02d", trait, rep, k_out, k_in))
        oof_inner[val_idx_inner, "SVR"] <- pred_svr$yhat[match(ids[val_idx_inner], pred_svr$ID)]
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="SVR", Seconds=proc.time()[3]-t0))

        # GAT
        t0 <- proc.time()[3]
        gat_out <- file.path("tmp", sprintf("pred_GAT_%s_rep%02d_out%02d_in%02d.csv", trait, rep, k_out, k_in))
        pred_gat <- call_py_model(GAT_SCRIPT, py$train_df, py$test_df, gat_out, python=PYTHON,
                                 tag=sprintf("GAT_%s_rep%02d_out%02d_in%02d", trait, rep, k_out, k_in))
        oof_inner[val_idx_inner, "GAT"] <- pred_gat$yhat[match(ids[val_idx_inner], pred_gat$ID)]
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="GAT", Seconds=proc.time()[3]-t0))
      }

      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="ALL", Seconds=proc.time()[3]-t_inner0))

      # Fit meta-learner on inner-OOF predictions (outer-train only)
      meta_train_rows <- train_idx_outer
      meta_ok <- meta_train_rows[rowSums(is.na(oof_inner[meta_train_rows, , drop=FALSE])) == 0]
      meta_ok <- meta_ok[!is.na(y[meta_ok])]

      if (length(meta_ok) < 5) {
        warning(sprintf("[%s rep%02d fold%02d] Not enough complete meta rows for stacking; skipping stacking.", trait, rep, k_out))
        stacker <- NULL
      } else {
        stacker <- fit_stack_ridge(oof_inner[meta_ok, , drop=FALSE], y[meta_ok])

        # log weights
        cf <- stacker$coef
        for (i in seq_len(nrow(cf))) {
          weight_log <- rbind(weight_log, data.table(
            Trait=trait, Rep=rep, OuterFold=k_out,
            Term=rownames(cf)[i], Coef=as.numeric(cf[i,1]), Lambda=stacker$lambda
          ))
        }
      }

      # ---------- Refit base models on full OUTER-TRAIN; predict OUTER-TEST ----------
      # base predictions on outer test
      base_te <- matrix(NA_real_, nrow=length(test_idx_outer), ncol=length(model_names),
                        dimnames=list(ids[test_idx_outer], model_names))

      # rrBLUP
      t0 <- proc.time()[3]
      base_te[, "rrBLUP"] <- predict_fold_rrBLUP(y, M, train_idx_outer, test_idx_outer)
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="rrBLUP", Seconds=proc.time()[3]-t0))

      # BayesB
      t0 <- proc.time()[3]
      base_te[, "BayesB"] <- predict_fold_BayesB(y, M, train_idx_outer, test_idx_outer,
                                                nIter=BAYESB_nIter, burnIn=BAYESB_burnIn, thin=BAYESB_thin)
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="BayesB", Seconds=proc.time()[3]-t0))

      # RKHS
      t0 <- proc.time()[3]
      base_te[, "RKHS"] <- predict_fold_RKHS(y, M, train_idx_outer, test_idx_outer)
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="RKHS", Seconds=proc.time()[3]-t0))

      # Python models on full outer train
      py_outer <- make_py_train_test(ids, M, y, train_idx_outer, test_idx_outer)

      # RF
      t0 <- proc.time()[3]
      rf_out <- file.path("tmp", sprintf("pred_RF_%s_rep%02d_out%02d_TEST.csv", trait, rep, k_out))
      pred_rf <- call_py_model(RF_SCRIPT, py_outer$train_df, py_outer$test_df, rf_out, python=PYTHON,
                              tag=sprintf("RF_%s_rep%02d_out%02d_TEST", trait, rep, k_out))
      base_te[, "RF"] <- pred_rf$yhat[match(ids[test_idx_outer], pred_rf$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="RF", Seconds=proc.time()[3]-t0))

      # SVR
      t0 <- proc.time()[3]
      svr_out <- file.path("tmp", sprintf("pred_SVR_%s_rep%02d_out%02d_TEST.csv", trait, rep, k_out))
      pred_svr <- call_py_model(SVR_SCRIPT, py_outer$train_df, py_outer$test_df, svr_out, python=PYTHON,
                               tag=sprintf("SVR_%s_rep%02d_out%02d_TEST", trait, rep, k_out))
      base_te[, "SVR"] <- pred_svr$yhat[match(ids[test_idx_outer], pred_svr$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="SVR", Seconds=proc.time()[3]-t0))

      # GAT
      t0 <- proc.time()[3]
      gat_out <- file.path("tmp", sprintf("pred_GAT_%s_rep%02d_out%02d_TEST.csv", trait, rep, k_out))
      pred_gat <- call_py_model(GAT_SCRIPT, py_outer$train_df, py_outer$test_df, gat_out, python=PYTHON,
                               tag=sprintf("GAT_%s_rep%02d_out%02d_TEST", trait, rep, k_out))
      base_te[, "GAT"] <- pred_gat$yhat[match(ids[test_idx_outer], pred_gat$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="GAT", Seconds=proc.time()[3]-t0))

      # ---------- Evaluate on OUTER TEST ----------
      y_te <- y[test_idx_outer]

      # Base models
      for (m in model_names) {
        met <- metrics_vec(y_te, base_te[, m])
        metrics_outer <- rbind(metrics_outer, data.table(
          Trait=trait, Rep=rep, OuterFold=k_out, Model=m,
          r=met$r, spearman=met$spearman, rmse=met$rmse, mae=met$mae
        ))
      }

      # Mean ensemble
      yhat_mean <- rowMeans(base_te, na.rm = FALSE)
      met_mean <- metrics_vec(y_te, yhat_mean)
      metrics_outer <- rbind(metrics_outer, data.table(
        Trait=trait, Rep=rep, OuterFold=k_out, Model="MeanEnsemble",
        r=met_mean$r, spearman=met_mean$spearman, rmse=met_mean$rmse, mae=met_mean$mae
      ))

      # Stacking
      if (!is.null(stacker)) {
        # stacker trained on inner-OOF; now apply to base predictions on outer test
        yhat_stack <- stacker$predict(base_te)
        met_stack <- metrics_vec(y_te, yhat_stack)
        metrics_outer <- rbind(metrics_outer, data.table(
          Trait=trait, Rep=rep, OuterFold=k_out, Model="Stacking",
          r=met_stack$r, spearman=met_stack$spearman, rmse=met_stack$rmse, mae=met_stack$mae
        ))
      } else {
        metrics_outer <- rbind(metrics_outer, data.table(
          Trait=trait, Rep=rep, OuterFold=k_out, Model="Stacking",
          r=NA_real_, spearman=NA_real_, rmse=NA_real_, mae=NA_real_
        ))
      }

      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TOTAL", Model="ALL", Seconds=proc.time()[3]-t_outer0))
    }
  }

  # ---------- Aggregate outer-CV results ----------
  # mean over (Rep, OuterFold) per model
  metrics_overall <- metrics_outer[, .(
    r       = mean(r, na.rm=TRUE),
    spearman= mean(spearman, na.rm=TRUE),
    rmse    = mean(rmse, na.rm=TRUE),
    mae     = mean(mae, na.rm=TRUE)
  ), by=.(Trait, Model)]

  fwrite(metrics_overall, file.path(outdir, "metrics", paste0("metrics_overall_", trait, ".tsv")), sep="\t")
  fwrite(metrics_outer,   file.path(outdir, "metrics", paste0("metrics_by_outer_", trait, ".tsv")), sep="\t")
  fwrite(weight_log,      file.path(outdir, "stacking", paste0("stacking_weights_", trait, ".tsv")), sep="\t")
  fwrite(runtime_log,     file.path(outdir, "runtime", paste0("runtime_by_model_", trait, ".tsv")), sep="\t")

  cat("[", trait, "] finished (nested CV)\n", sep="")
}

cat("\n=== NESTED PIPELINE FINISHED SUCCESSFULLY ===\n")

