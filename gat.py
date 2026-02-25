# -*- coding: utf-8 -*-
# py_models/gat.py


import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GATRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim: int = 32, heads: int = 2, dropout: float = 0.1):
        """
        Lightweight graph attention network for regression. The hidden dimension,
        number of heads and dropout have been reduced to accelerate training
        without changing the underlying GAT method. Smaller hidden_dim and
        fewer attention heads lead to faster computation per epoch.
        """
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        # The second GAT layer uses a single head and concatenation disabled.
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.lin  = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.lin(x).squeeze(-1)
        return x


def build_knn_edge_index(X: np.ndarray, k: int = 10) -> torch.Tensor:
    """Build an undirected KNN graph (k neighbors per node)."""
    n = X.shape[0]
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, idx = nbrs.kneighbors(X)

    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:(k + 1)].reshape(-1)

    edge = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(edge, dtype=torch.long)


def train_gat(
    data: Data,
    train_mask: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 123,
    patience: int = 10,
    min_delta: float = 1e-4
) -> nn.Module:
    """
    Train a graph attention regressor with optional early stopping.

    Parameters
    ----------
    data : Data
        Combined graph containing node features and edges.
    train_mask : torch.Tensor
        Boolean mask indicating which nodes have labels (used for training).
    epochs : int
        Maximum number of training epochs. Reduced from 300 to 100 to
        shorten runtime without changing the fundamental method.
    lr : float
        Learning rate for Adam optimizer.
    weight_decay : float
        Weight decay for Adam optimizer.
    seed : int
        Random seed for reproducibility.
    patience : int
        Number of epochs to wait for an improvement before stopping early.
    min_delta : float
        Minimum relative improvement in loss to reset patience counter.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Always use CPU; GPU is disabled because it is unavailable in this environment.
    device = torch.device("cpu")
    data = data.to(device)
    train_mask = train_mask.to(device)

    model = GATRegressor(in_dim=data.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track best loss for early stopping
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

        # Early stopping: check relative improvement
        curr_loss = loss.item()
        if curr_loss < best_loss - min_delta:
            best_loss = curr_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Stop training early
                break

    return model.cpu()


def main(
    train_path,
    test_path,
    out_path,
    k: int = 10,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 123,
    patience: int = 10,
    min_delta: float = 1e-4
):

    graph_mode = os.environ.get("GRAPH_MODE", "transductive").strip().lower()

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    if "ID" not in train.columns or "y" not in train.columns:
        raise ValueError("train.csv must contain columns: ID, y, <features>")
    if "ID" not in test.columns:
        raise ValueError("test.csv must contain column: ID")

    Xtr = train.drop(columns=["ID", "y"]).values
    ytr = train["y"].values.astype(np.float32)
    Xte = test.drop(columns=["ID"]).values

    # scaler fit on TRAIN only (no leakage)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    if graph_mode == "separate":
        # Old behavior (ablation only): separate graphs for train and test
        edge_tr = build_knn_edge_index(Xtr_s, k=k)
        edge_te = build_knn_edge_index(Xte_s, k=k)

        data_tr = Data(
            x=torch.tensor(Xtr_s, dtype=torch.float32),
            edge_index=edge_tr,
            y=torch.tensor(ytr, dtype=torch.float32)
        )
        train_mask = torch.ones(data_tr.x.shape[0], dtype=torch.bool)
        model = train_gat(
            data_tr,
            train_mask,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            patience=patience,
            min_delta=min_delta,
        )

        data_te = Data(
            x=torch.tensor(Xte_s, dtype=torch.float32),
            edge_index=edge_te,
            y=torch.zeros(Xte_s.shape[0], dtype=torch.float32)
        )
        with torch.no_grad():
            pred_te = model(data_te.x, data_te.edge_index).numpy()

    else:
        # Recommended: transductive graph on train+test nodes
        X_all = np.vstack([Xtr_s, Xte_s])
        edge_all = build_knn_edge_index(X_all, k=k)

        y_all = np.concatenate([ytr, np.full((Xte_s.shape[0],), np.nan, dtype=np.float32)])
        y_all_t = torch.tensor(np.nan_to_num(y_all, nan=0.0), dtype=torch.float32)

        data_all = Data(
            x=torch.tensor(X_all, dtype=torch.float32),
            edge_index=edge_all,
            y=y_all_t
        )
        train_mask = torch.zeros(X_all.shape[0], dtype=torch.bool)
        train_mask[:Xtr_s.shape[0]] = True

        model = train_gat(
            data_all,
            train_mask,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            patience=patience,
            min_delta=min_delta,
        )

        with torch.no_grad():
            pred_all = model(data_all.x, data_all.edge_index).numpy()
        pred_te = pred_all[Xtr_s.shape[0]:]

    out = pd.DataFrame({"ID": test["ID"].values, "yhat": pred_te})
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python gat.py train.csv test.csv pred.csv")
    main(sys.argv[1], sys.argv[2], sys.argv[3])

