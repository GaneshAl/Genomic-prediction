#!/usr/bin/env Rscript
# ==============================================================
# Fully Nested CV Genomic Prediction Pipeline
#
# Base learners: rrBLUP + BayesB + RKHS + RF + SVR + GAT
#
# Outer CV: unbiased performance estimate
# Inner CV: generate OOF base predictions on outer-train set
#           fit meta-learner ONLY on those OOF rows
#
# Includes (outer-test evaluation):
#  - Base model metrics
#  - Mean ensemble
#  - Properly nested stacking (ridge)
#  - Component ablations of the stacking framework:
#      * Leave-one-base-model-out (FULL vs NO_<model>)
#      * Meta-learner type (Ridge vs OLS vs ElasticNet)
#      * Scaling of meta-features (scale vs no-scale)
#      * Non-nested (optimistic) stacking comparator
#      * Optional global-preprocessing (leaky) comparator (clearly labeled)
#  - Stacking weight logs
#  - Runtime logs
# ==============================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript run_driver_nested_fixed.R <phenotype_file>\n")
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

# Optional: also compute a clearly-labeled leaky comparator (global X preprocessing)
DO_LEAKY_GLOBAL_PREPROC_COMPARATOR <- TRUE

dir.create("results_nested", showWarnings = FALSE, recursive = TRUE)
dir.create("tmp", showWarnings = FALSE, recursive = TRUE)

# ---------------- SOURCE MODEL HELPERS ----------------
# NOTE: For RKHS, use the FIXED rkhs.R that accepts marker matrix M and
# computes distances internally (see models_RKHS_fixed.R).
source("models/rrblup.R")
source("models/bayesb.R")
source("models/rkhs.R")
source("utils/call_python_model.R")

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

make_folds <- function(idx, K) {
  sample(rep(seq_len(K), length.out = length(idx)))
}

# Foldwise preprocessing: train-only mean imputation + centering
# Also does foldwise zero-variance filtering (on train) to avoid using test info.
preprocess_fold <- function(M_raw, train_idx, test_idx) {
  # train-only variance filter
  v <- apply(M_raw[train_idx, , drop=FALSE], 2, var, na.rm = TRUE)
  keep <- which(is.finite(v) & v > 0)
  if (length(keep) < 2) stop("Too few non-zero-variance markers in training fold.")

  Mtr <- M_raw[train_idx, keep, drop=FALSE]
  Mte <- M_raw[test_idx,  keep, drop=FALSE]

  cm <- colMeans(Mtr, na.rm = TRUE)
  # impute
  for (j in seq_len(ncol(Mtr))) {
    if (anyNA(Mtr[,j])) Mtr[is.na(Mtr[,j]), j] <- cm[j]
    if (anyNA(Mte[,j])) Mte[is.na(Mte[,j]), j] <- cm[j]
  }
  # center using train means AFTER imputation (equals cm)
  Mtr <- sweep(Mtr, 2, cm, "-")
  Mte <- sweep(Mte, 2, cm, "-")
  list(Mtr=as.matrix(Mtr), Mte=as.matrix(Mte), keep=keep, center=cm)
}

# Global (leaky) preprocessing comparator (for demonstration only)
preprocess_global_leaky <- function(M_raw, train_idx, test_idx) {
  v <- apply(M_raw, 2, var, na.rm = TRUE)
  keep <- which(is.finite(v) & v > 0)
  M <- M_raw[, keep, drop=FALSE]
  cm <- colMeans(M, na.rm = TRUE)
  for (j in seq_len(ncol(M))) {
    if (anyNA(M[,j])) M[is.na(M[,j]), j] <- cm[j]
  }
  M <- sweep(M, 2, cm, "-")
  list(Mtr=as.matrix(M[train_idx, , drop=FALSE]), Mte=as.matrix(M[test_idx, , drop=FALSE]))
}

make_py_train_test <- function(ids, Mtr, Mte, ytr) {
  train_df <- data.table(ID = ids, y = ytr)
  test_df  <- data.table(ID = ids)
  train_df <- cbind(train_df, as.data.table(Mtr))
  test_df  <- cbind(test_df,  as.data.table(Mte))
  list(train_df=train_df, test_df=test_df)
}

fit_meta <- function(X, y, method=c("ridge","ols","enet"), do_scale=TRUE, alpha_enet=0.5) {
  method <- match.arg(method)
  if (do_scale) {
    Xs <- scale(X)
    ctr <- attr(Xs, "scaled:center")
    scl <- attr(Xs, "scaled:scale")
    scl[scl == 0] <- 1
    scale_with <- function(A) sweep(sweep(A, 2, ctr, "-"), 2, scl, "/")
  } else {
    Xs <- as.matrix(X)
    ctr <- rep(0, ncol(X)); names(ctr) <- colnames(X)
    scl <- rep(1, ncol(X)); names(scl) <- colnames(X)
    scale_with <- function(A) as.matrix(A)
  }

  if (method == "ols") {
    df <- data.frame(y=y, Xs)
    fit <- lm(y ~ . , data=df)
    predict_fun <- function(newX) {
      nd <- data.frame(scale_with(newX))
      colnames(nd) <- colnames(Xs)
      as.numeric(predict(fit, newdata=nd))
    }
    coef_vec <- coef(fit)
    return(list(method=method, lambda=NA_real_, coef=matrix(coef_vec, ncol=1, dimnames=list(names(coef_vec),"coef")),
                predict=predict_fun, scale_center=ctr, scale_scale=scl))
  }

  alpha <- if (method == "ridge") 0 else alpha_enet
  cv <- cv.glmnet(x = Xs, y = y, alpha = alpha)
  list(
    method=method,
    cv=cv,
    lambda=cv$lambda.min,
    coef=as.matrix(coef(cv, s="lambda.min")),
    predict=function(newX) as.numeric(predict(cv, scale_with(newX), s="lambda.min")),
    scale_center=ctr,
    scale_scale=scl
  )
}

# ---------------- READ GENOTYPES ----------------
cat("Reading genotypes.\n")
geno <- fread("GP_MARKERS.raw")
ids_all <- geno$IID
M_raw <- as.matrix(geno[, -(1:6), with = FALSE])
rownames(M_raw) <- ids_all
colnames(M_raw) <- colnames(geno)[-(1:6)]
rm(geno); gc()

# ---------------- READ PHENOTYPES ----------------
pheno <- fread(pheno_file)
if (!(ID_COL %in% names(pheno))) stop("Phenotype file must have ID column: ", ID_COL)
setkeyv(pheno, ID_COL)
pheno <- pheno[J(ids_all), nomatch=0]
if (nrow(pheno) == 0) stop("No overlapping IDs between genotype and phenotype file.")

pheno_ids <- pheno[[ID_COL]]
ord <- match(pheno_ids, ids_all)
ids <- ids_all[ord]
M_raw <- M_raw[ord, , drop=FALSE]

model_names <- c("rrBLUP", "BayesB", "RKHS", "RF", "SVR", "GAT")

# ---------------- MAIN LOOP ----------------
for (trait in TRAITS) {
  cat("\n=== Trait:", trait, "===\n")
  if (!(trait %in% names(pheno))) {
    cat("Trait not found in phenotype file:", trait, " -- skipping\n")
    next
  }

  y <- as.numeric(pheno[[trait]])
  n <- length(y)
  idx_nonNA <- which(!is.na(y))

  outdir <- file.path("results_nested", trait)
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir, "metrics"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir, "stacking"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir, "runtime"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir, "ablation"), showWarnings = FALSE, recursive = TRUE)

  metrics_outer <- data.table(Trait=character(), Rep=integer(), OuterFold=integer(), Model=character(),
                              r=numeric(), spearman=numeric(), rmse=numeric(), mae=numeric())

  # Framework ablations: stacking variants (outer-test)
  framework_outer <- data.table(Trait=character(), Rep=integer(), OuterFold=integer(), Condition=character(),
                                r=numeric(), spearman=numeric(), rmse=numeric(), mae=numeric())

  # Leave-one-base-model-out (ridge, scaled, properly nested)
  ablation_outer <- data.table(Trait=character(), Rep=integer(), OuterFold=integer(), Condition=character(),
                               r=numeric(), spearman=numeric(), rmse=numeric(), mae=numeric(),
                               delta_r=numeric(), delta_rmse=numeric(), delta_mae=numeric())

  weight_log <- data.table(Trait=character(), Rep=integer(), OuterFold=integer(), Condition=character(),
                           Term=character(), Coef=numeric(), Lambda=numeric())

  runtime_log <- data.table(Trait=character(), Rep=integer(), OuterFold=integer(), Phase=character(), Model=character(), Seconds=numeric())

  for (rep in seq_len(N_REPS)) {
    folds_outer <- make_folds(idx_nonNA, K_OUTER)

    for (k_out in seq_len(K_OUTER)) {
      t_outer0 <- proc.time()[3]
      test_idx_outer  <- idx_nonNA[folds_outer == k_out]
      train_idx_outer <- setdiff(idx_nonNA, test_idx_outer)

      # Foldwise preprocessing for outer fold (NO leakage)
      pf_outer <- preprocess_fold(M_raw, train_idx_outer, test_idx_outer)
      Mtr_outer <- pf_outer$Mtr
      Mte_outer <- pf_outer$Mte

      # ---------- INNER CV to create OOF predictions on OUTER-TRAIN ----------
      t_inner0 <- proc.time()[3]
      folds_inner <- make_folds(train_idx_outer, K_INNER)
      oof_inner <- matrix(NA_real_, nrow=n, ncol=length(model_names), dimnames=list(ids, model_names))

      for (k_in in seq_len(K_INNER)) {
        val_idx_inner <- train_idx_outer[folds_inner == k_in]
        fit_idx_inner <- setdiff(train_idx_outer, val_idx_inner)

        # Preprocess within INNER split (still within outer-train)
        pf_in <- preprocess_fold(M_raw, fit_idx_inner, val_idx_inner)
        M_fit <- pf_in$Mtr
        M_val <- pf_in$Mte

        # rrBLUP
        y_local <- c(y[fit_idx_inner], y[val_idx_inner])
        M_local <- rbind(M_fit, M_val)
        n_fit <- nrow(M_fit); n_val <- nrow(M_val)
        t0 <- proc.time()[3]
        oof_inner[val_idx_inner, "rrBLUP"] <- predict_fold_rrBLUP(y_local, M_local, seq_len(n_fit), (n_fit+1):(n_fit+n_val))
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="rrBLUP", Seconds=proc.time()[3]-t0))

        # BayesB
        t0 <- proc.time()[3]
        oof_inner[val_idx_inner, "BayesB"] <- predict_fold_BayesB(y_local, M_local, seq_len(n_fit), (n_fit+1):(n_fit+n_val),
                                                                  nIter=BAYESB_nIter, burnIn=BAYESB_burnIn, thin=BAYESB_thin)
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="BayesB", Seconds=proc.time()[3]-t0))

        # RKHS (fixed)
        t0 <- proc.time()[3]
        oof_inner[val_idx_inner, "RKHS"] <- predict_fold_RKHS(y_local, M_local,
                                                              train_idx=seq_len(n_fit),
                                                              test_idx=(n_fit+1):(n_fit+n_val))
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="RKHS", Seconds=proc.time()[3]-t0))

        # Python models
        py <- make_py_train_test(ids[fit_idx_inner], M_fit, M_val, y[fit_idx_inner])

        t0 <- proc.time()[3]
        rf_out <- file.path("tmp", sprintf("pred_RF_%s_rep%02d_out%02d_in%02d.csv", trait, rep, k_out, k_in))
        pred_rf <- call_py_model(RF_SCRIPT, py$train_df, data.table(ID=ids[val_idx_inner], as.data.table(M_val)), rf_out, python=PYTHON,
                                 tag=sprintf("RF_%s_rep%02d_out%02d_in%02d", trait, rep, k_out, k_in))
        oof_inner[val_idx_inner, "RF"] <- pred_rf$yhat[match(ids[val_idx_inner], pred_rf$ID)]
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="RF", Seconds=proc.time()[3]-t0))

        t0 <- proc.time()[3]
        svr_out <- file.path("tmp", sprintf("pred_SVR_%s_rep%02d_out%02d_in%02d.csv", trait, rep, k_out, k_in))
        pred_svr <- call_py_model(SVR_SCRIPT, py$train_df, data.table(ID=ids[val_idx_inner], as.data.table(M_val)), svr_out, python=PYTHON,
                                  tag=sprintf("SVR_%s_rep%02d_out%02d_in%02d", trait, rep, k_out, k_in))
        oof_inner[val_idx_inner, "SVR"] <- pred_svr$yhat[match(ids[val_idx_inner], pred_svr$ID)]
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="SVR", Seconds=proc.time()[3]-t0))

        t0 <- proc.time()[3]
        gat_out <- file.path("tmp", sprintf("pred_GAT_%s_rep%02d_out%02d_in%02d.csv", trait, rep, k_out, k_in))
        pred_gat <- call_py_model(GAT_SCRIPT, py$train_df, data.table(ID=ids[val_idx_inner], as.data.table(M_val)), gat_out, python=PYTHON,
                                  tag=sprintf("GAT_%s_rep%02d_out%02d_in%02d", trait, rep, k_out, k_in),
                                  env=c("GRAPH_MODE=transductive"))
        oof_inner[val_idx_inner, "GAT"] <- pred_gat$yhat[match(ids[val_idx_inner], pred_gat$ID)]
        runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="GAT", Seconds=proc.time()[3]-t0))
      }

      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="INNER", Model="ALL", Seconds=proc.time()[3]-t_inner0))

      # Fit meta on COMPLETE OOF rows within outer-train
      meta_rows <- train_idx_outer
      meta_ok <- meta_rows[rowSums(is.na(oof_inner[meta_rows, , drop=FALSE])) == 0]
      meta_ok <- meta_ok[!is.na(y[meta_ok])]

      # ---------- Refit base models on full OUTER-TRAIN and predict OUTER-TEST ----------
      y_te <- y[test_idx_outer]
      base_te <- matrix(NA_real_, nrow=length(test_idx_outer), ncol=length(model_names),
                        dimnames=list(ids[test_idx_outer], model_names))

      # Prepare local y/M for R base models (avoid index mismatch)
      y_local_outer <- c(y[train_idx_outer], y[test_idx_outer])
      M_local_outer <- rbind(Mtr_outer, Mte_outer)
      ntr <- nrow(Mtr_outer); nte <- nrow(Mte_outer)

      # rrBLUP
      t0 <- proc.time()[3]
      base_te[,"rrBLUP"] <- predict_fold_rrBLUP(y_local_outer, M_local_outer,
                                               train_idx=seq_len(ntr),
                                               test_idx=(ntr+1):(ntr+nte))
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="rrBLUP", Seconds=proc.time()[3]-t0))

      # BayesB
      t0 <- proc.time()[3]
      base_te[,"BayesB"] <- predict_fold_BayesB(y_local_outer, M_local_outer,
                                               train_idx=seq_len(ntr),
                                               test_idx=(ntr+1):(ntr+nte),
                                               nIter=BAYESB_nIter, burnIn=BAYESB_burnIn, thin=BAYESB_thin)
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="BayesB", Seconds=proc.time()[3]-t0))

      # RKHS (fixed)
      t0 <- proc.time()[3]
      base_te[,"RKHS"] <- predict_fold_RKHS(y_local_outer, M_local_outer,
                                           train_idx=seq_len(ntr),
                                           test_idx=(ntr+1):(ntr+nte))
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="RKHS", Seconds=proc.time()[3]-t0))

      # Python models on full outer train
      train_df_outer <- data.table(ID=ids[train_idx_outer], y=y[train_idx_outer])
      test_df_outer  <- data.table(ID=ids[test_idx_outer])
      train_df_outer <- cbind(train_df_outer, as.data.table(Mtr_outer))
      test_df_outer  <- cbind(test_df_outer,  as.data.table(Mte_outer))

      t0 <- proc.time()[3]
      rf_out <- file.path("tmp", sprintf("pred_RF_%s_rep%02d_out%02d_TEST.csv", trait, rep, k_out))
      pred_rf <- call_py_model(RF_SCRIPT, train_df_outer, test_df_outer, rf_out, python=PYTHON,
                               tag=sprintf("RF_%s_rep%02d_out%02d_TEST", trait, rep, k_out))
      base_te[,"RF"] <- pred_rf$yhat[match(ids[test_idx_outer], pred_rf$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="RF", Seconds=proc.time()[3]-t0))

      t0 <- proc.time()[3]
      svr_out <- file.path("tmp", sprintf("pred_SVR_%s_rep%02d_out%02d_TEST.csv", trait, rep, k_out))
      pred_svr <- call_py_model(SVR_SCRIPT, train_df_outer, test_df_outer, svr_out, python=PYTHON,
                                tag=sprintf("SVR_%s_rep%02d_out%02d_TEST", trait, rep, k_out))
      base_te[,"SVR"] <- pred_svr$yhat[match(ids[test_idx_outer], pred_svr$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="SVR", Seconds=proc.time()[3]-t0))

      t0 <- proc.time()[3]
      gat_out <- file.path("tmp", sprintf("pred_GAT_%s_rep%02d_out%02d_TEST.csv", trait, rep, k_out))
      pred_gat <- call_py_model(GAT_SCRIPT, train_df_outer, test_df_outer, gat_out, python=PYTHON,
                                tag=sprintf("GAT_%s_rep%02d_out%02d_TEST", trait, rep, k_out),
                                env=c("GRAPH_MODE=transductive"))
      base_te[,"GAT"] <- pred_gat$yhat[match(ids[test_idx_outer], pred_gat$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TRAIN", Model="GAT", Seconds=proc.time()[3]-t0))

      # ---------- Base metrics ----------
      for (m in model_names) {
        met <- metrics_vec(y_te, base_te[,m])
        metrics_outer <- rbind(metrics_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Model=m,
                                                        r=met$r, spearman=met$spearman, rmse=met$rmse, mae=met$mae))
      }

      # Mean ensemble
      yhat_mean <- rowMeans(base_te)
      met_mean <- metrics_vec(y_te, yhat_mean)
      metrics_outer <- rbind(metrics_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Model="MeanEnsemble",
                                                      r=met_mean$r, spearman=met_mean$spearman, rmse=met_mean$rmse, mae=met_mean$mae))

      # ---------- Stacking variants (framework ablations) ----------
      if (length(meta_ok) >= 5) {
        X_meta <- oof_inner[meta_ok, , drop=FALSE]
        y_meta <- y[meta_ok]

        # Properly nested ridge (scaled)
        st_ridge <- fit_meta(X_meta, y_meta, method="ridge", do_scale=TRUE)
        yhat_stack <- st_ridge$predict(base_te)
        met_stack <- metrics_vec(y_te, yhat_stack)
        metrics_outer <- rbind(metrics_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Model="Stacking_Ridge",
                                                        r=met_stack$r, spearman=met_stack$spearman, rmse=met_stack$rmse, mae=met_stack$mae))
        framework_outer <- rbind(framework_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="NESTED_RIDGE_SCALE",
                                                            r=met_stack$r, spearman=met_stack$spearman, rmse=met_stack$rmse, mae=met_stack$mae))

        # Log weights
        cf <- st_ridge$coef
        for (i in seq_len(nrow(cf))) {
          weight_log <- rbind(weight_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="NESTED_RIDGE_SCALE",
                                                    Term=rownames(cf)[i], Coef=as.numeric(cf[i,1]), Lambda=st_ridge$lambda))
        }

        # Meta-learner ablation: OLS
        st_ols <- fit_meta(X_meta, y_meta, method="ols", do_scale=TRUE)
        yhat_ols <- st_ols$predict(base_te)
        met_ols <- metrics_vec(y_te, yhat_ols)
        framework_outer <- rbind(framework_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="NESTED_OLS_SCALE",
                                                            r=met_ols$r, spearman=met_ols$spearman, rmse=met_ols$rmse, mae=met_ols$mae))

        # Meta-learner ablation: Elastic Net
        st_en <- fit_meta(X_meta, y_meta, method="enet", do_scale=TRUE, alpha_enet=0.5)
        yhat_en <- st_en$predict(base_te)
        met_en <- metrics_vec(y_te, yhat_en)
        framework_outer <- rbind(framework_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="NESTED_ENET05_SCALE",
                                                            r=met_en$r, spearman=met_en$spearman, rmse=met_en$rmse, mae=met_en$mae))

        # Scaling ablation: ridge without scaling
        st_ridge_ns <- fit_meta(X_meta, y_meta, method="ridge", do_scale=FALSE)
        yhat_ns <- st_ridge_ns$predict(base_te)
        met_ns <- metrics_vec(y_te, yhat_ns)
        framework_outer <- rbind(framework_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="NESTED_RIDGE_NOSCALE",
                                                            r=met_ns$r, spearman=met_ns$spearman, rmse=met_ns$rmse, mae=met_ns$mae))

        # Non-nested (optimistic) stacking comparator:
        # fit base models on outer-train, predict outer-train, fit meta on those (leaky wrt base model fitting)
        # This is included ONLY as a bias demonstration.
        # We approximate by using outer-train predictions produced by refitting each base model once.
        # (To keep runtime reasonable, we reuse outer-train fitted models via a quick re-run.)
        #
        # Compute base predictions on outer-train itself
        base_tr <- matrix(NA_real_, nrow=length(train_idx_outer), ncol=length(model_names),
                          dimnames=list(ids[train_idx_outer], model_names))
        n_tr <- nrow(Mtr_outer)
        # rrBLUP
        y_tr_only <- y[train_idx_outer]
        base_tr[,"rrBLUP"] <- predict_fold_rrBLUP(y_tr_only, Mtr_outer, seq_len(n_tr), seq_len(n_tr))
        # BayesB
        base_tr[,"BayesB"] <- predict_fold_BayesB(y_tr_only, Mtr_outer, seq_len(n_tr), seq_len(n_tr),
                                                 nIter=BAYESB_nIter, burnIn=BAYESB_burnIn, thin=BAYESB_thin)
        # RKHS
        base_tr[,"RKHS"] <- predict_fold_RKHS(y_tr_only, Mtr_outer, seq_len(n_tr), seq_len(n_tr))
        # Python models
        rf_tr_out <- file.path("tmp", sprintf("pred_RF_%s_rep%02d_out%02d_TRAIN.csv", trait, rep, k_out))
        pred_rf_tr <- call_py_model(RF_SCRIPT, train_df_outer, cbind(data.table(ID=ids[train_idx_outer]), as.data.table(Mtr_outer)), rf_tr_out,
                                    python=PYTHON, tag=sprintf("RF_%s_rep%02d_out%02d_TRAIN", trait, rep, k_out))
        base_tr[,"RF"] <- pred_rf_tr$yhat[match(ids[train_idx_outer], pred_rf_tr$ID)]

        svr_tr_out <- file.path("tmp", sprintf("pred_SVR_%s_rep%02d_out%02d_TRAIN.csv", trait, rep, k_out))
        pred_svr_tr <- call_py_model(SVR_SCRIPT, train_df_outer, cbind(data.table(ID=ids[train_idx_outer]), as.data.table(Mtr_outer)), svr_tr_out,
                                     python=PYTHON, tag=sprintf("SVR_%s_rep%02d_out%02d_TRAIN", trait, rep, k_out))
        base_tr[,"SVR"] <- pred_svr_tr$yhat[match(ids[train_idx_outer], pred_svr_tr$ID)]

        gat_tr_out <- file.path("tmp", sprintf("pred_GAT_%s_rep%02d_out%02d_TRAIN.csv", trait, rep, k_out))
        pred_gat_tr <- call_py_model(GAT_SCRIPT, train_df_outer, cbind(data.table(ID=ids[train_idx_outer]), as.data.table(Mtr_outer)), gat_tr_out,
                                     python=PYTHON, tag=sprintf("GAT_%s_rep%02d_out%02d_TRAIN", trait, rep, k_out),
                                     env=c("GRAPH_MODE=transductive"))
        base_tr[,"GAT"] <- pred_gat_tr$yhat[match(ids[train_idx_outer], pred_gat_tr$ID)]

        st_nonnested <- fit_meta(base_tr, y[train_idx_outer], method="ridge", do_scale=TRUE)
        yhat_nonnested <- st_nonnested$predict(base_te)
        met_nonnested <- metrics_vec(y_te, yhat_nonnested)
        framework_outer <- rbind(framework_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="NONNESTED_RIDGE_OPTIMISTIC",
                                                            r=met_nonnested$r, spearman=met_nonnested$spearman, rmse=met_nonnested$rmse, mae=met_nonnested$mae))

        # Optional leaky global preprocessing comparator (clearly labeled)
        if (isTRUE(DO_LEAKY_GLOBAL_PREPROC_COMPARATOR)) {
          pf_leak <- preprocess_global_leaky(M_raw, train_idx_outer, test_idx_outer)
          # quick: evaluate rrBLUP only to show magnitude; extend if you want
          y_local_leak <- c(y[train_idx_outer], y[test_idx_outer])
          yhat_rr_leak <- predict_fold_rrBLUP(y_local_leak, rbind(pf_leak$Mtr, pf_leak$Mte),
                                              train_idx=seq_len(nrow(pf_leak$Mtr)),
                                              test_idx=(nrow(pf_leak$Mtr)+1):(nrow(pf_leak$Mtr)+nrow(pf_leak$Mte)))
          met_rr_leak <- metrics_vec(y_te, yhat_rr_leak)
          framework_outer <- rbind(framework_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="LEAKY_GLOBAL_PREPROC_rrBLUP_ONLY",
                                                              r=met_rr_leak$r, spearman=met_rr_leak$spearman, rmse=met_rr_leak$rmse, mae=met_rr_leak$mae))
        }

        # Leave-one-base-model-out ablation (nested ridge, scaled)
        ablation_outer <- rbind(ablation_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition="FULL",
                                                          r=met_stack$r, spearman=met_stack$spearman, rmse=met_stack$rmse, mae=met_stack$mae,
                                                          delta_r=0, delta_rmse=0, delta_mae=0))
        for (m_drop in model_names) {
          keep <- setdiff(model_names, m_drop)
          st_drop <- fit_meta(X_meta[, keep, drop=FALSE], y_meta, method="ridge", do_scale=TRUE)
          yhat_drop <- st_drop$predict(base_te[, keep, drop=FALSE])
          met_drop <- metrics_vec(y_te, yhat_drop)
          ablation_outer <- rbind(ablation_outer, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Condition=paste0("NO_", m_drop),
                                                            r=met_drop$r, spearman=met_drop$spearman, rmse=met_drop$rmse, mae=met_drop$mae,
                                                            delta_r=met_stack$r - met_drop$r,
                                                            delta_rmse=met_drop$rmse - met_stack$rmse,
                                                            delta_mae=met_drop$mae - met_stack$mae))
        }
      } else {
        warning(sprintf("[%s rep%02d out%02d] Not enough complete meta rows; stacking skipped.", trait, rep, k_out))
      }

      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, OuterFold=k_out, Phase="OUTER_TOTAL", Model="ALL", Seconds=proc.time()[3]-t_outer0))
    }
  }

  # ---------- Aggregate results ----------
  metrics_overall <- metrics_outer[, .(
    r        = mean(r, na.rm=TRUE),
    spearman = mean(spearman, na.rm=TRUE),
    rmse     = mean(rmse, na.rm=TRUE),
    mae      = mean(mae, na.rm=TRUE)
  ), by=.(Trait, Model)]

  framework_summary <- framework_outer[, .(
    r        = mean(r, na.rm=TRUE),
    spearman = mean(spearman, na.rm=TRUE),
    rmse     = mean(rmse, na.rm=TRUE),
    mae      = mean(mae, na.rm=TRUE)
  ), by=.(Trait, Condition)]

  ablation_summary <- ablation_outer[Condition != "FULL", .(
    mean_delta_r    = mean(delta_r, na.rm=TRUE),
    mean_delta_rmse = mean(delta_rmse, na.rm=TRUE),
    mean_delta_mae  = mean(delta_mae, na.rm=TRUE),
    n              = .N
  ), by=.(Trait, Condition)]

  # ---------- Write outputs ----------
  fwrite(metrics_overall,   file.path(outdir, "metrics",  paste0("metrics_overall_", trait, ".tsv")), sep="\t")
  fwrite(metrics_outer,     file.path(outdir, "metrics",  paste0("metrics_by_outer_", trait, ".tsv")), sep="\t")
  fwrite(framework_outer,   file.path(outdir, "ablation", paste0("framework_ablation_by_outer_", trait, ".tsv")), sep="\t")
  fwrite(framework_summary, file.path(outdir, "ablation", paste0("framework_ablation_summary_", trait, ".tsv")), sep="\t")
  fwrite(ablation_outer,    file.path(outdir, "ablation", paste0("leave1out_ablation_by_outer_", trait, ".tsv")), sep="\t")
  fwrite(ablation_summary,  file.path(outdir, "ablation", paste0("leave1out_ablation_summary_", trait, ".tsv")), sep="\t")
  fwrite(weight_log,        file.path(outdir, "stacking", paste0("stacking_weights_", trait, ".tsv")), sep="\t")
  fwrite(runtime_log,       file.path(outdir, "runtime",  paste0("runtime_by_model_", trait, ".tsv")), sep="\t")

  cat("[", trait, "] finished (nested CV + framework ablations)\n", sep="")
}

cat("\n=== NESTED PIPELINE FINISHED SUCCESSFULLY ===\n")

