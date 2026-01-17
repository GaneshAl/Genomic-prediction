# py_models/svr.py
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

    # Scale using TRAIN only (pipeline handles this correctly)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))
    ])

    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    out = pd.DataFrame({"ID": test["ID"].values, "yhat": yhat})
    out.to_csv(out_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python svr.py train.csv test.csv pred.csv")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
