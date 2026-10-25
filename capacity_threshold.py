"""
capacity_threshold.py   —   Week 8 experiments

Two things left over from the last check-in:

    PART 1  REDO THE COST COMPARISON ON CLEAN FEATURES
            Last week I found my model's top feature was PolicyNumber, a claim ID.
            So every number I reported before that was partly built on identifiers.
            This re-runs the same four approaches with the ID and calendar columns gone.

    PART 2  PUT THE TWO LEVERS TOGETHER
            The cost-optimal threshold flags 586 claims, which no investigations
            unit could work. So instead of asking "what threshold minimises cost",
            ask "given we can review K claims, what does that buy us".


"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

SEED = 42
FN_COST = 10
FP_COST = 1

# the nine columns that are identifiers or calendar fields, not claim facts
ADMIN = ["PolicyNumber", "RepNumber", "Year", "Month", "WeekOfMonth", "DayOfWeek",
         "MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed"]

df = pd.read_csv("data/carclaims.csv")
df["FraudFound"] = df["FraudFound"].str.strip().map({"Yes": 1, "No": 0})
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["FraudFound"] + ADMIN)
y = df["FraudFound"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

n_fraud = int(y_te.sum())
print(f"{len(df):,} claims, {X.shape[1]} features after dropping {len(ADMIN)} ID/calendar columns")
print(f"test set: {len(y_te):,} claims, {n_fraud} fraudulent ({100*y_te.mean():.1f}%)\n")


def forest(**kw):
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, **kw)


def report(name, pred):
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    cost = FN_COST * fn + FP_COST * fp
    prec = tp / (tp + fp) if tp + fp else 0
    print(f"{name:<22}{recall_score(y_te, pred):<10.3f}{prec:<12.3f}"
          f"{f'{tp}/{n_fraud}':<12}{fp:<14}{cost:<8}")
    return cost


# =====================================================================
print("=" * 78)
print("PART 1   SAME FOUR APPROACHES, CLEAN FEATURES")
print("=" * 78)
print(f"{'Approach':<22}{'Recall':<10}{'Precision':<12}{'Caught':<12}{'False alarms':<14}{'Cost':<8}")
print("-" * 78)

c1 = report("1. Baseline", forest().fit(X_tr, y_tr).predict(X_te))

X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
c2 = report("2. SMOTE", forest().fit(X_sm, y_sm).predict(X_te))

c3 = report("3. Class weights",
            forest(class_weight={0: FP_COST, 1: FN_COST}).fit(X_tr, y_tr).predict(X_te))

# threshold picked on training out-of-fold predictions only
oof = cross_val_predict(forest(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
grid = np.linspace(0.01, 0.60, 60)
costs = []
for t in grid:
    tn, fp, fn, tp = confusion_matrix(y_tr, (oof >= t).astype(int)).ravel()
    costs.append(FN_COST * fn + FP_COST * fp)
best_t = grid[int(np.argmin(costs))]

model = forest().fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
c4 = report("4. Cost threshold", (proba >= best_t).astype(int))

print("-" * 78)
print(f"\nThreshold chosen on training data only: {best_t:.3f}")
print(f"Cost threshold vs SMOTE: {c4 - c2:+} ({100*(c4-c2)/c2:+.1f}%)\n")


# =====================================================================
print("=" * 78)
print("PART 2   WHAT A REAL REVIEW CAPACITY BUYS YOU")
print("=" * 78)
print("Review the top K claims by fraud probability. K is the team's weekly capacity.\n")
print(f"{'Capacity K':<14}{'Cutoff':<10}{'Caught':<12}{'Precision':<12}{'% fraud found':<16}{'Cost':<8}")
print("-" * 78)

order = np.argsort(-proba)
y_sorted = y_te.values[order]
p_sorted = proba[order]

for k in [100, 200, 300, 500, 1000]:
    tp = int(y_sorted[:k].sum())
    fp = k - tp
    fn = n_fraud - tp
    print(f"{k:<14}{p_sorted[k-1]:<10.3f}{f'{tp}/{n_fraud}':<12}"
          f"{tp/k:<12.3f}{100*tp/n_fraud:<16.1f}{FN_COST*fn + fp:<8}")

# the unconstrained cost-optimal point, for comparison
flagged = int((proba >= best_t).sum())
tp = int(((proba >= best_t) & (y_te.values == 1)).sum())
print(f"{'unconstrained':<14}{best_t:<10.3f}{f'{tp}/{n_fraud}':<12}"
      f"{tp/flagged:<12.3f}{100*tp/n_fraud:<16.1f}{FN_COST*(n_fraud-tp) + (flagged-tp):<8}"
      f"  <- flags {flagged} claims")

print(f"\nBase rate if you reviewed claims at random: {y_te.mean():.3f}")
print(f"Lift at K=100: {int(y_sorted[:100].sum())/100 / y_te.mean():.1f}x")
