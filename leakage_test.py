"""
leakage_test.py

Same model, same data, same seed. The only difference is the order of two steps.

    PAPER ORDER   balance the whole dataset, then split   (what the paper does)
    CORRECT ORDER split, then balance training only       (honest evaluation)

Run:  python leakage_test.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

SEED = 42

# ---- load and encode ----
df = pd.read_csv("data/carclaims.csv")
df["FraudFound"] = df["FraudFound"].str.strip().map({"Yes": 1, "No": 0})
for col in df.columns:
    if df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["FraudFound"])
y = df["FraudFound"]
print(f"{len(df):,} claims, {int(y.sum())} fraudulent ({100*y.mean():.1f}%)\n")


def run(order):
    if order == "paper":
        Xb, yb = SMOTE(random_state=SEED).fit_resample(X, y)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xb, yb, test_size=0.2, stratify=yb, random_state=SEED)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED)
        X_tr, y_tr = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    return accuracy_score(y_te, pred), recall_score(y_te, pred), 100 * y_te.mean()


for label, order in [("PAPER ORDER  (balance, then split)", "paper"),
                     ("CORRECT ORDER (split, then balance)", "correct")]:
    acc, rec, rate = run(order)
    print(f"{label}")
    print(f"   test set is {rate:.0f}% fraud")
    print(f"   accuracy {acc:.4f}   recall {rec:.4f}\n")