# py_models/gat.py
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# PyTorch Geometric
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
    k = min(k, n-1)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    _, idx = nbrs.kneighbors(X)

    # idx includes self at position 0, so skip it
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:(k+1)].reshape(-1)

    # Make undirected edges by adding reverse
    edge = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(edge, dtype=torch.long)

def main(train_path, test_path, out_path,
         k=15, epochs=300, lr=1e-3, weight_decay=1e-4, seed=123):

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    if "ID" not in train.columns or "y" not in train.columns:
        raise ValueError("train.csv must contain columns: ID, y, <features...>")
    if "ID" not in test.columns:
        raise ValueError("test.csv must contain column: ID")

    # Combine fold train+test for graph construction
    Xtr = train.drop(columns=["ID", "y"]).values
    ytr = train["y"].values.astype(np.float32)

    Xte = test.drop(columns=["ID"]).values

    # Scale using TRAIN only to avoid leakage
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    X = np.vstack([Xtr_s, Xte_s]).astype(np.float32)
    n_tr = Xtr_s.shape[0]
    n_te = Xte_s.shape[0]
    n = n_tr + n_te

    edge_index = build_knn_edge_index(X, k=k)

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.empty(n, dtype=torch.float32)
    y[:] = float("nan")
    y[:n_tr] = torch.tensor(ytr, dtype=torch.float32)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_tr] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[n_tr:] = True

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GATRegressor(in_dim=data.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Simple training loop (no val split here; add if you want early stopping)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).detach().cpu().numpy()

    yhat_test = pred[n_tr:]

    out = pd.DataFrame({"ID": test["ID"].values, "yhat": yhat_test})
    out.to_csv(out_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python gat.py train.csv test.csv pred.csv")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
