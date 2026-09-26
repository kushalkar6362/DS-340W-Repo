"""
cost_sensitive.py

Parent paper 1 handles class imbalance by inventing synthetic fraud cases (SMOTE).
Parent paper 2 argues you should weigh the two error types by what they cost instead.

This compares four approaches on the same data, same model, same seed:

    1. BASELINE          no balancing at all
    2. SMOTE             invent synthetic fraud cases in the training fold
    3. CLASS WEIGHTS     tell the forest a fraud is worth 10x during training
    4. COST THRESHOLD    keep the model, move the decision cutoff to minimise cost

Scored with parent paper 2's cost function:   expected cost = 10 x FN + 1 x FP

The threshold in (4) is chosen on out-of-fold predictions from the TRAINING set
only. The test set is never used to pick it.

Run:  python cost_sensitive.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

SEED = 42
FN_COST = 10   # a missed fraud is paid out in full
FP_COST = 1    # a false alarm wastes an investigator

# ---- load and encode ----
df = pd.read_csv("data/carclaims.csv")
df["FraudFound"] = df["FraudFound"].str.strip().map({"Yes": 1, "No": 0})
for col in df.columns:
    if df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["FraudFound"])
y = df["FraudFound"]

# ---- split FIRST, the way parent paper 2 does it ----
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

print(f"{len(df):,} claims, {int(y.sum())} fraudulent ({100*y.mean():.1f}%)")
print(f"test set: {len(y_te):,} claims, {int(y_te.sum())} fraudulent")
print(f"cost function: {FN_COST} x missed fraud  +  {FP_COST} x false alarm\n")


def forest(**kw):
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, **kw)


def report(name, pred):
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    cost = FN_COST * fn + FP_COST * fp
    print(f"{name:<20}"
          f"recall {recall_score(y_te, pred):.3f}   "
          f"precision {precision_score(y_te, pred, zero_division=0):.3f}   "
          f"caught {tp:>3}/{tp+fn:<5}"
          f"false alarms {fp:>4}   "
          f"COST {cost:>5}")
    return cost


print("-" * 104)

# 1 — baseline
m = forest().fit(X_tr, y_tr)
c1 = report("1. Baseline", m.predict(X_te))

# 2 — SMOTE on the training fold only
X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
c2 = report("2. SMOTE", forest().fit(X_sm, y_sm).predict(X_te))

# 3 — class weights during training
c3 = report("3. Class weights",
            forest(class_weight={0: FP_COST, 1: FN_COST}).fit(X_tr, y_tr).predict(X_te))

# 4 — pick the decision threshold that minimises cost, using TRAINING data only
oof = cross_val_predict(forest(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
grid = np.linspace(0.01, 0.60, 60)
costs = []
for t in grid:
    tn, fp, fn, tp = confusion_matrix(y_tr, (oof >= t).astype(int)).ravel()
    costs.append(FN_COST * fn + FP_COST * fp)
best_t = grid[int(np.argmin(costs))]

proba_te = forest().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
c4 = report(f"4. Cost threshold", (proba_te >= best_t).astype(int))

print("-" * 104)
print(f"\nThreshold chosen on training data only: {best_t:.3f}  (default is 0.500)")

results = {"baseline": c1, "SMOTE": c2, "class weights": c3, "cost threshold": c4}
winner = min(results, key=results.get)
print(f"Lowest expected cost: {winner} at {results[winner]}")
print(f"Versus SMOTE: {results[winner] - c2:+} ({100*(results[winner]-c2)/c2:+.1f}%)")