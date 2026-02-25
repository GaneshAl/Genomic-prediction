# -*- coding: utf-8 -*-
# py_models/rf.py
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def main(train_path, test_path, out_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    if "ID" not in train.columns or "y" not in train.columns:
        raise ValueError("train.csv must contain columns: ID, y, <features...>")
    if "ID" not in test.columns:
        raise ValueError("test.csv must contain column: ID")

    X_train = train.drop(columns=["ID", "y"])
    y_train = train["y"].values
    X_test  = test.drop(columns=["ID"])

    # Reasonable defaults for high-dimension SNP data; tune later.
    # Reduce the number of trees to speed up training.  More trees usually
    # improve accuracy at the cost of slower learning; lowering this value
    # speeds up the random forest.  See accompanying documentation for
    # details on trade-offs between tree count and runtime.
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=123,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    out = pd.DataFrame({"ID": test["ID"].values, "yhat": yhat})
    out.to_csv(out_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python rf.py train.csv test.csv pred.csv")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
