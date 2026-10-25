"""
capacity_and_ratio.py   —   Week 6 experiments

Three things peer reviewers and parent paper 3 each pushed on:

    PART 1  COST RATIO SWEEP
            Last check-in picked a decision threshold using a 10-to-1 cost ratio.
            Is 10 doing the work, or does the result hold across ratios?

    PART 2  FIXED REVIEW CAPACITY
            An investigations unit can only work so many claims a week.
            So rank every claim by fraud probability and only look at the top K.
            This is precision@K, and it is what a carrier actually cares about.

    PART 3  WHICH FEATURES DRIVE THE SCORE
            Parent paper 3 used SHAP, LIME and ELI5 and concluded the top drivers
            are fault, policyholder age, vehicle age, base policy and deductible.
            Do I get the same answer on the same dataset?


"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42

# ---- load and encode ----
df = pd.read_csv("data/carclaims.csv")
df["FraudFound"] = df["FraudFound"].str.strip().map({"Yes": 1, "No": 0})
feature_names = [c for c in df.columns if c != "FraudFound"]
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):   # works on pandas 2.x and 3.x
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["FraudFound"])
y = df["FraudFound"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)


def forest():
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)


# fit once, reuse everywhere
model = forest().fit(X_tr, y_tr)
proba_te = model.predict_proba(X_te)[:, 1]

# out-of-fold training probabilities, used to pick thresholds
# the test set is never used to choose anything
oof = cross_val_predict(forest(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]

n_fraud_te = int(y_te.sum())
print(f"test set: {len(y_te):,} claims, {n_fraud_te} fraudulent "
      f"({100*y_te.mean():.1f}%)\n")


# =====================================================================
# PART 1 — does the result depend on the 10-to-1 ratio?
# =====================================================================
print("=" * 78)
print("PART 1   COST RATIO SWEEP")
print("=" * 78)
print(f"{'ratio':<10}{'threshold':<12}{'caught':<14}{'false alarms':<15}{'cost':<10}")
print("-" * 78)

grid = np.linspace(0.01, 0.60, 60)

for fn_cost in [2, 5, 10, 15, 20]:
    # choose threshold on TRAINING out-of-fold predictions only
    costs = []
    for t in grid:
        tn, fp, fn, tp = confusion_matrix(y_tr, (oof >= t).astype(int)).ravel()
        costs.append(fn_cost * fn + 1 * fp)
    best_t = grid[int(np.argmin(costs))]

    # apply it to the test set
    tn, fp, fn, tp = confusion_matrix(y_te, (proba_te >= best_t).astype(int)).ravel()
    print(f"{fn_cost:>2}-to-1   {best_t:<12.3f}"
          f"{f'{tp}/{tp+fn}':<14}{fp:<15}{fn_cost*fn + fp:<10}")

print("\nIf the threshold moves smoothly with the ratio and never jumps,")
print("the approach is not tuned to one arbitrary number.\n")


# =====================================================================
# PART 2 — precision at a fixed weekly review capacity
# =====================================================================
print("=" * 78)
print("PART 2   PRECISION AT FIXED REVIEW CAPACITY")
print("=" * 78)
print("Rank all test claims by fraud probability, review only the top K.\n")
print(f"{'capacity K':<14}{'frauds found':<16}{'precision@K':<16}{'% of all fraud':<16}")
print("-" * 78)

order = np.argsort(-proba_te)          # highest probability first
y_sorted = y_te.values[order]

for k in [50, 100, 200, 300, 500, 1000]:
    found = int(y_sorted[:k].sum())
    print(f"{k:<14}{found:<16}{found/k:<16.3f}{100*found/n_fraud_te:<16.1f}")

base = y_te.mean()
print(f"\nBase rate if you picked claims at random: {base:.3f}")
print(f"Lift at K=100: {int(y_sorted[:100].sum())/100 / base:.1f}x better than random\n")


# =====================================================================
# PART 3 — what is actually driving the score
# =====================================================================
print("=" * 78)
print("PART 3   TOP FEATURES  (compare against parent paper 3)")
print("=" * 78)

paper3 = {"Fault", "AgeOfPolicyHolder", "AgeOfVehicle", "BasePolicy", "Deductible"}


def show(imp, label):
    print(f"\n{label}")
    print(f"{'rank':<7}{'feature':<26}{'importance':<14}{'named by paper 3?'}")
    print("-" * 78)
    for i, (name, val) in enumerate(imp.head(10).items(), 1):
        print(f"{i:<7}{name:<26}{val:<14.4f}{'yes' if name in paper3 else ''}")
    hits = [f for f in imp.head(10).index if f in paper3]
    print(f"-> {len(hits)} of paper 3's 5 named features in my top 10")


# --- 3a: every column, the way I had it ---
show(pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False),
     "3a. ALL COLUMNS")

# --- 3b: drop administrative and calendar fields, the way paper 3 does ---
ADMIN = ["PolicyNumber", "RepNumber", "Year", "Month", "WeekOfMonth", "DayOfWeek",
         "MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed"]

X2_tr, X2_te = X_tr.drop(columns=ADMIN), X_te.drop(columns=ADMIN)
model2 = forest().fit(X2_tr, y_tr)
proba2_te = model2.predict_proba(X2_te)[:, 1]

show(pd.Series(model2.feature_importances_, index=X2_tr.columns).sort_values(ascending=False),
     f"3b. {len(ADMIN)} ADMIN / CALENDAR COLUMNS DROPPED")

# does dropping them cost anything at a realistic capacity?
y2_sorted = y_te.values[np.argsort(-proba2_te)]
print(f"\nprecision@100   all columns: {int(y_sorted[:100].sum())/100:.3f}"
      f"   admin dropped: {int(y2_sorted[:100].sum())/100:.3f}")
