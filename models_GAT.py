# py_models/gat.py
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


def build_knn_edge_index(X, k=15):
    # X: (n, p) numpy
    n = X.shape[0]
    if n <= 1:
        # degenerate
        return torch.empty((2, 0), dtype=torch.long)

    k = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, idx = nbrs.kneighbors(X)

    # idx includes self at position 0, so skip it
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:(k + 1)].reshape(-1)

    # Make undirected edges by adding reverse
    edge = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(edge, dtype=torch.long)


def main(train_path, test_path, out_path,
         k=15, epochs=300, lr=1e-3, weight_decay=1e-4, seed=123):

    np.random.seed(seed)
    torch.manual_seed(seed)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    if "ID" not in train.columns or "y" not in train.columns:
        raise ValueError("train.csv must contain columns: ID, y, <features.>")
    if "ID" not in test.columns:
        raise ValueError("test.csv must contain column: ID")

    Xtr = train.drop(columns=["ID", "y"]).values
    ytr = train["y"].values.astype(np.float32)

    Xte = test.drop(columns=["ID"]).values

    # strict: scaler fit on TRAIN only
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    # strict: graph built separately
    edge_tr = build_knn_edge_index(Xtr_s, k=k)
    edge_te = build_knn_edge_index(Xte_s, k=k)

    data_tr = Data(
        x=torch.tensor(Xtr_s, dtype=torch.float32),
        edge_index=edge_tr,
        y=torch.tensor(ytr, dtype=torch.float32)
    )
    data_te = Data(
        x=torch.tensor(Xte_s, dtype=torch.float32),
        edge_index=edge_te
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tr = data_tr.to(device)
    data_te = data_te.to(device)

    model = GATRegressor(in_dim=data_tr.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(data_tr.x, data_tr.edge_index)
        loss = F.mse_loss(pred, data_tr.y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_te = model(data_te.x, data_te.edge_index).detach().cpu().numpy()

    out = pd.DataFrame({"ID": test["ID"].values, "yhat": pred_te})
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python gat.py train.csv test.csv pred.csv")
    main(sys.argv[1], sys.argv[2], sys.argv[3])

