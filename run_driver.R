#!/usr/bin/env Rscript
# ==============================================================
# Genomic Prediction Pipeline (Methods paper version)
# rrBLUP + BayesB + RKHS + RF + SVR + GAT
# + OOF predictions
# + Mean ensemble + Ridge stacking
# + Architecture ablation (meta-only, leave-one-out)
# + Weight stability across CV reps
# + Residual complementarity (residual correlation)
# ==============================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript run_driver.R <phenotype_file>\n")
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

K_FOLDS <- 5
N_REPS  <- 10
set.seed(123)

# BayesB / RKHS iterations (final)
BAYESB_nIter  <- 12000
BAYESB_burnIn <- 4000
BAYESB_thin   <- 5

RKHS_nIter  <- 6000
RKHS_burnIn <- 2000
RKHS_thin   <- 5

# Python scripts
PYTHON <- "python"
RF_SCRIPT  <- "py_models/rf.py"
SVR_SCRIPT <- "py_models/svr.py"
GAT_SCRIPT <- "py_models/gat.py"

dir.create("results", showWarnings = FALSE, recursive = TRUE)
dir.create("tmp", showWarnings = FALSE, recursive = TRUE)

# ---------------- SOURCE MODEL HELPERS ----------------
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

fit_stack_ridge <- function(X, y) {
  # X already OOF; we scale predictors for stability
  Xs <- scale(X)
  cv <- cv.glmnet(x = Xs, y = y, alpha = 0)
  list(
    cv = cv,
    lambda = cv$lambda.min,
    coef = as.matrix(coef(cv, s="lambda.min")),
    predict = function(newX) as.numeric(predict(cv, scale(newX), s="lambda.min"))
  )
}

# Meta-only ablation: refit ridge leaving one column out
ablate_stack <- function(oof_X, y, model_names) {
  valid <- which(!is.na(y) & rowSums(is.na(oof_X)) == 0)
  Xv <- oof_X[valid, , drop=FALSE]
  yv <- y[valid]
  
  out <- list()
  
  # FULL
  st_full <- fit_stack_ridge(Xv, yv)
  yhat_full <- st_full$predict(Xv)
  m_full <- metrics_vec(yv, yhat_full)
  out[["FULL"]] <- list(metrics = m_full, weights = st_full$coef, lambda = st_full$lambda)
  
  # Leave-one-out
  for (m in model_names) {
    keep <- setdiff(model_names, m)
    st <- fit_stack_ridge(Xv[, keep, drop=FALSE], yv)
    yhat <- st$predict(Xv[, keep, drop=FALSE])
    met <- metrics_vec(yv, yhat)
    out[[paste0("NO_", m)]] <- list(metrics = met, weights = st$coef, lambda = st$lambda, keep=keep)
  }
  out
}

# ---------------- READ GENOTYPES ----------------
cat("Reading genotypes...\n")
geno <- fread("GP_MARKERS.raw")
ids <- geno$IID

M <- as.matrix(geno[, -(1:6), with = FALSE])
rownames(M) <- ids
colnames(M) <- colnames(geno)[-(1:6)]
rm(geno); gc()

# Remove zero-variance SNPs
v <- apply(M, 2, var, na.rm = TRUE)
M <- M[, v > 0, drop = FALSE]

# Mean impute + center
cm <- colMeans(M, na.rm = TRUE)
for (j in seq_len(ncol(M))) {
  idx <- is.na(M[, j])
  if (any(idx)) M[idx, j] <- cm[j]
}
M <- scale(M, center = TRUE, scale = FALSE)

# ---------------- PRECOMPUTE RKHS KERNEL ONCE ----------------
cat("Precomputing RKHS kernel...\n")
D  <- as.matrix(dist(M))
h  <- median(D[upper.tri(D)])
Kk <- exp(-(D^2) / (2*h^2))

# ---------------- READ PHENOTYPES ----------------
cat("Reading phenotypes...\n")
pheno <- fread(pheno_file)
pheno <- pheno[match(ids, pheno[[ID_COL]]), ]
stopifnot(all(pheno[[ID_COL]] == ids))

model_names <- c("rrBLUP","BayesB","RKHS","RF","SVR","GAT")

# ---------------- TRAIT LOOP ----------------
for (trait in TRAITS) {
  
  cat("\n==============================\n")
  cat("Trait:", trait, "\n")
  cat("==============================\n")
  
  outdir <- file.path("results", paste0("Trait_", trait))
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir,"oof"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir,"metrics"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir,"stacking"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir,"ablation"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir,"runtime"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(outdir,"complementarity"), showWarnings = FALSE, recursive = TRUE)
  
  y <- pheno[[trait]]
  idx_nonNA <- which(!is.na(y))
  n <- length(y)
  
  preds_all <- array(
    NA_real_,
    dim = c(n, length(model_names), N_REPS),
    dimnames = list(ids, model_names, paste0("rep", 1:N_REPS))
  )
  
  # runtime log: long table
  runtime_log <- data.table(
    Trait=character(), Rep=integer(), Fold=integer(),
    Model=character(), Seconds=numeric()
  )
  
  # stacking weights per rep (weight stability)
  weight_log <- data.table(
    Trait=character(), Rep=integer(), Model=character(),
    Weight=numeric(), Lambda=numeric()
  )
  
  # rep-wise metrics for base and ensembles
  metrics_rep <- data.table(
    Trait=character(), Rep=integer(), Model=character(),
    r=numeric(), spearman=numeric(), rmse=numeric(), mae=numeric()
  )
  
  # -------------- CV repeats --------------
  for (rep in seq_len(N_REPS)) {
    
    cat("  CV repeat", rep, "\n")
    
    folds <- sample(rep(1:K_FOLDS, length.out = length(idx_nonNA)))
    
    for (k in seq_len(K_FOLDS)) {
      
      test_idx  <- idx_nonNA[folds == k]
      train_idx <- setdiff(idx_nonNA, test_idx)
      
      # Prepare fold dataframes for python models (use centered M)
      train_df <- data.frame(ID = ids[train_idx], y = y[train_idx], M[train_idx, , drop=FALSE])
      test_df  <- data.frame(ID = ids[test_idx],  M[test_idx,  , drop=FALSE])
      
      # -------- rrBLUP --------
      t0 <- proc.time()[3]
      preds_all[test_idx, "rrBLUP", rep] <- predict_fold_rrBLUP(y, M, train_idx, test_idx)
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, Fold=k, Model="rrBLUP", Seconds=proc.time()[3]-t0))
      
      # -------- BayesB --------
      t0 <- proc.time()[3]
      preds_all[test_idx, "BayesB", rep] <- predict_fold_BayesB(
        y, M, train_idx, test_idx,
        nIter=BAYESB_nIter, burnIn=BAYESB_burnIn, thin=BAYESB_thin
      )
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, Fold=k, Model="BayesB", Seconds=proc.time()[3]-t0))
      
      # -------- RKHS --------
      t0 <- proc.time()[3]
      preds_all[test_idx, "RKHS", rep] <- predict_fold_RKHS(
        y, Kk, train_idx, test_idx,
        nIter=RKHS_nIter, burnIn=RKHS_burnIn, thin=RKHS_thin
      )
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, Fold=k, Model="RKHS", Seconds=proc.time()[3]-t0))
      
      # -------- RF (python) --------
      t0 <- proc.time()[3]
      pred_rf <- call_py_model(RF_SCRIPT, train_df, test_df,
                               out_pred=file.path("tmp", "pred_rf.csv"),
                               python=PYTHON)
      preds_all[test_idx, "RF", rep] <- pred_rf$yhat[match(ids[test_idx], pred_rf$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, Fold=k, Model="RF", Seconds=proc.time()[3]-t0))
      
      # -------- SVR (python) --------
      t0 <- proc.time()[3]
      pred_svr <- call_py_model(SVR_SCRIPT, train_df, test_df,
                                out_pred=file.path("tmp", "pred_svr.csv"),
                                python=PYTHON)
      preds_all[test_idx, "SVR", rep] <- pred_svr$yhat[match(ids[test_idx], pred_svr$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, Fold=k, Model="SVR", Seconds=proc.time()[3]-t0))
      
      # -------- GAT (python) --------
      t0 <- proc.time()[3]
      pred_gat <- call_py_model(GAT_SCRIPT, train_df, test_df,
                                out_pred=file.path("tmp", "pred_gat.csv"),
                                python=PYTHON)
      preds_all[test_idx, "GAT", rep] <- pred_gat$yhat[match(ids[test_idx], pred_gat$ID)]
      runtime_log <- rbind(runtime_log, data.table(Trait=trait, Rep=rep, Fold=k, Model="GAT", Seconds=proc.time()[3]-t0))
    }
    
    # ---------- Rep-wise evaluation (base + ensembles + stacking weights) ----------
    # Base predictions for this rep
    oof_rep <- preds_all[, , rep, drop=FALSE][,,1]
    valid <- which(!is.na(y) & rowSums(is.na(oof_rep)) == 0)
    
    # Base model metrics
    for (m in model_names) {
      met <- metrics_vec(y[valid], oof_rep[valid, m])
      metrics_rep <- rbind(metrics_rep, data.table(
        Trait=trait, Rep=rep, Model=m,
        r=met$r, spearman=met$spearman, rmse=met$rmse, mae=met$mae
      ))
    }
    
    # Mean ensemble metrics
    yhat_mean <- rowMeans(oof_rep[valid, , drop=FALSE])
    met_mean <- metrics_vec(y[valid], yhat_mean)
    metrics_rep <- rbind(metrics_rep, data.table(
      Trait=trait, Rep=rep, Model="MeanEnsemble",
      r=met_mean$r, spearman=met_mean$spearman, rmse=met_mean$rmse, mae=met_mean$mae
    ))
    
    # Ridge stacking (per rep) + weight log
    st <- fit_stack_ridge(oof_rep[valid, , drop=FALSE], y[valid])
    yhat_stack <- st$predict(oof_rep[valid, , drop=FALSE])
    met_stack <- metrics_vec(y[valid], yhat_stack)
    metrics_rep <- rbind(metrics_rep, data.table(
      Trait=trait, Rep=rep, Model="Stacking",
      r=met_stack$r, spearman=met_stack$spearman, rmse=met_stack$rmse, mae=met_stack$mae
    ))
    
    # Store weights (exclude intercept row 1)
    coef_vec <- st$coef
    # coef matrix has rownames like "(Intercept)", "rrBLUP", ...
    rn <- rownames(coef_vec)
    for (i in seq_along(rn)) {
      if (rn[i] == "(Intercept)") next
      weight_log <- rbind(weight_log, data.table(
        Trait=trait, Rep=rep, Model=rn[i],
        Weight=as.numeric(coef_vec[i,1]),
        Lambda=st$lambda
      ))
    }
  } # end reps
  
  # -------------- Aggregate OOF across reps (your original logic) --------------
  oof_mean <- apply(preds_all, c(1,2), mean, na.rm = TRUE)
  
  saveRDS(preds_all,  file.path(outdir,"oof", paste0("OOF_full_", trait, ".rds")))
  saveRDS(oof_mean,   file.path(outdir,"oof", paste0("OOF_mean_", trait, ".rds")))
  saveRDS(list(ids=ids, y=y), file.path(outdir,"oof", paste0("y_ids_", trait, ".rds")))
  
  # -------------- Overall metrics using OOF_mean --------------
  valid_all <- which(!is.na(y) & rowSums(is.na(oof_mean)) == 0)
  
  metrics_overall <- data.table(Trait=trait, Model=character(), r=numeric(), spearman=numeric(), rmse=numeric(), mae=numeric())
  for (m in model_names) {
    met <- metrics_vec(y[valid_all], oof_mean[valid_all, m])
    metrics_overall <- rbind(metrics_overall, data.table(Trait=trait, Model=m, r=met$r, spearman=met$spearman, rmse=met$rmse, mae=met$mae))
  }
  
  # Mean ensemble on OOF_mean
  yhat_mean_all <- rowMeans(oof_mean[valid_all, , drop=FALSE])
  met_mean_all <- metrics_vec(y[valid_all], yhat_mean_all)
  metrics_overall <- rbind(metrics_overall, data.table(Trait=trait, Model="MeanEnsemble", r=met_mean_all$r, spearman=met_mean_all$spearman, rmse=met_mean_all$rmse, mae=met_mean_all$mae))
  
  # Stacking on OOF_mean
  st_all <- fit_stack_ridge(oof_mean[valid_all, , drop=FALSE], y[valid_all])
  yhat_stack_all <- st_all$predict(oof_mean[valid_all, , drop=FALSE])
  met_stack_all <- metrics_vec(y[valid_all], yhat_stack_all)
  metrics_overall <- rbind(metrics_overall, data.table(Trait=trait, Model="Stacking", r=met_stack_all$r, spearman=met_stack_all$spearman, rmse=met_stack_all$rmse, mae=met_stack_all$mae))
  
  fwrite(metrics_overall, file.path(outdir,"metrics", paste0("metrics_overall_", trait, ".tsv")), sep="\t")
  fwrite(metrics_rep,     file.path(outdir,"metrics", paste0("metrics_by_rep_", trait, ".tsv")), sep="\t")
  fwrite(runtime_log,     file.path(outdir,"runtime", paste0("runtime_by_model_", trait, ".tsv")), sep="\t")
  fwrite(weight_log,      file.path(outdir,"stacking", paste0("stacking_weights_by_rep_", trait, ".tsv")), sep="\t")
  
  # -------------- Weight stability summary --------------
  # mean/sd/CV + top-1 frequency
  wsum <- weight_log[, .(
    mean_w = mean(Weight, na.rm=TRUE),
    sd_w   = sd(Weight, na.rm=TRUE),
    cv_w   = ifelse(abs(mean(Weight, na.rm=TRUE)) < 1e-12, NA_real_, sd(Weight, na.rm=TRUE)/abs(mean(Weight, na.rm=TRUE))),
    q025 = quantile(Weight, 0.025, na.rm=TRUE),
    q975 = quantile(Weight, 0.975, na.rm=TRUE)
  ), by=.(Trait, Model)]
  
  # top-1 frequency per trait
  top1 <- weight_log[, .SD[which.max(Weight)], by=.(Trait, Rep)]
  top1_freq <- top1[, .(top1_freq = .N / N_REPS), by=.(Trait, Model)]
  wstab <- merge(wsum, top1_freq, by=c("Trait","Model"), all.x=TRUE)
  fwrite(wstab, file.path(outdir,"stacking", paste0("weight_stability_", trait, ".tsv")), sep="\t")
  
  # -------------- Architecture ablation (meta-only) --------------
  abl <- ablate_stack(oof_mean, y, model_names)
  ablation_tbl <- data.table(
    Trait=trait, Condition=character(),
    r=numeric(), spearman=numeric(), rmse=numeric(), mae=numeric(),
    delta_r=numeric(), delta_rmse=numeric(), delta_mae=numeric()
  )
  
  full_m <- abl[["FULL"]]$metrics
  for (cond in names(abl)) {
    met <- abl[[cond]]$metrics
    ablation_tbl <- rbind(ablation_tbl, data.table(
      Trait=trait, Condition=cond,
      r=met$r, spearman=met$spearman, rmse=met$rmse, mae=met$mae,
      delta_r = full_m$r - met$r,
      delta_rmse = met$rmse - full_m$rmse,
      delta_mae  = met$mae  - full_m$mae
    ))
  }
  fwrite(ablation_tbl, file.path(outdir,"ablation", paste0("ablation_meta_only_", trait, ".tsv")), sep="\t")
  
  # -------------- Residual complementarity (OOF_mean residual correlations) --------------
  # residuals = y - yhat_model (valid rows only)
  Rmat <- matrix(NA_real_, nrow=length(model_names), ncol=length(model_names),
                 dimnames=list(model_names, model_names))
  resid_mat <- sapply(model_names, function(m) y[valid_all] - oof_mean[valid_all, m])
  for (i in seq_along(model_names)) {
    for (j in seq_along(model_names)) {
      Rmat[i,j] <- safe_cor(resid_mat[,i], resid_mat[,j])
    }
  }
  saveRDS(Rmat, file.path(outdir,"complementarity", paste0("residual_corr_", trait, ".rds")))
  fwrite(as.data.table(Rmat, keep.rownames="Model"),
         file.path(outdir,"complementarity", paste0("residual_corr_", trait, ".tsv")), sep="\t")
  
  # -------------- Save overall stacking weights on OOF_mean --------------
  coef_all <- st_all$coef
  coef_dt <- data.table(Trait=trait, Term=rownames(coef_all), Coef=as.numeric(coef_all[,1]), Lambda=st_all$lambda)
  fwrite(coef_dt, file.path(outdir,"stacking", paste0("stacking_coef_overall_", trait, ".tsv")), sep="\t")
  
  cat("[", trait, "] finished\n", sep="")
}

cat("\n=== PIPELINE FINISHED SUCCESSFULLY ===\n")
