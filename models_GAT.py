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
    def __init__(self, in_dim, hidden_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
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


def build_knn_edge_index(X: np.ndarray, k: int = 15) -> torch.Tensor:
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


def train_gat(data: Data,
              train_mask: torch.Tensor,
              epochs: int = 300,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              seed: int = 123) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    train_mask = train_mask.to(device)

    model = GATRegressor(in_dim=data.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

    return model.cpu()


def main(train_path, test_path, out_path,
         k=15, epochs=300, lr=1e-3, weight_decay=1e-4, seed=123):

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
        model = train_gat(data_tr, train_mask, epochs=epochs, lr=lr, weight_decay=weight_decay, seed=seed)

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

        model = train_gat(data_all, train_mask, epochs=epochs, lr=lr, weight_decay=weight_decay, seed=seed)

        with torch.no_grad():
            pred_all = model(data_all.x, data_all.edge_index).numpy()
        pred_te = pred_all[Xtr_s.shape[0]:]

    out = pd.DataFrame({"ID": test["ID"].values, "yhat": pred_te})
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python gat.py train.csv test.csv pred.csv")
    main(sys.argv[1], sys.argv[2], sys.argv[3])

