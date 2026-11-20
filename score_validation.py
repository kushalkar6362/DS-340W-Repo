"""
score_validation.py

The final evaluation. This is the only script that reads splits/clean/validation.csv.

That partition has been held out since week 3. No training, no threshold tuning and
no model selection has ever touched it. Everything reported here is a single pass.

The validation set is 1,542 claims against 3,084 in test, so every budget is halved
to keep the review rate equivalent. A budget of 100 on test is 50 here.

PART 1  the four models at the equivalent budget
PART 2  the cost-optimal budget, the same derivation as marginal_value.py
PART 3  ranking against thresholding, the headline comparison

Run once:  python score_validation.py
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

SEED = 42
FN_COST = 10
FP_COST = 1
TARGET = "FraudFound"


def load(name):
    df = pd.read_csv(f"splits/clean/{name}.csv")
    return df.drop(columns=[TARGET]), df[TARGET]


X_tr, y_tr = load("train")
X_te, y_te = load("test")
X_va, y_va = load("validation")          # first and only read

n_fraud = int(y_va.sum())
n_val = len(y_va)
base = y_va.mean()

# keep the review rate the same as a budget of 100 on the test set
BUDGET = int(round(100 * n_val / len(y_te)))

print("=" * 80)
print("FINAL EVALUATION ON THE HELD-OUT VALIDATION SET")
print("=" * 80)
print(f"validation: {n_val:,} claims, {n_fraud} fraud ({100*base:.1f}%)")
print(f"equivalent review budget: {BUDGET} claims "
      f"(same review rate as 100 of {len(y_te):,} on test)\n")


def rf(**kw):
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **kw)


def hgb():
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                          learning_rate=0.06)


MODELS = {
    "Random forest": rf(),
    "RF + calibration": CalibratedClassifierCV(rf(), method="isotonic", cv=5),
    "Gradient boosting": hgb(),
    "GB + calibration": CalibratedClassifierCV(hgb(), method="isotonic", cv=5),
}

print("fitting on the training split only...\n")
proba, cum = {}, {}
for name, m in MODELS.items():
    p = m.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    proba[name] = p
    ranked = y_va.values[np.argsort(-p)]
    cum[name] = np.concatenate([[0], np.cumsum(ranked)])


# ================= PART 1 =================
print("=" * 80)
print(f"PART 1   PRECISION AT THE EQUIVALENT BUDGET (K = {BUDGET})")
print("=" * 80)
print(f"{'Model':<22}{'Caught':<13}{'Precision':<13}{'Fraud loss':<13}{'Lift'}")
print("-" * 80)

for name in MODELS:
    hits = int(cum[name][BUDGET])
    print(f"{name:<22}{f'{hits}/{n_fraud}':<13}{hits/BUDGET:<13.3f}"
          f"{BUDGET-hits:<13}{hits/BUDGET/base:.1f}x")
print("-" * 80)
print(f"\nreviewing {BUDGET} claims at random would give precision {base:.3f}")


# ================= PART 2 =================
def best_budget(hits, fn_cost):
    k = np.arange(len(hits))
    cost = fn_cost * (n_fraud - hits) + FP_COST * (k - hits)
    b = int(np.argmin(cost))
    return b, int(cost[b]), int(hits[b])


print("\n" + "=" * 80)
print("PART 2   COST-OPTIMAL REVIEW BUDGET  (10-to-1 ratio)")
print("=" * 80)
print(f"{'Model':<22}{'Best budget':<15}{'Caught':<13}{'Precision':<13}{'Cost'}")
print("-" * 80)

for name in MODELS:
    k, cost, caught = best_budget(cum[name], 10)
    prec = caught / k if k else 0
    print(f"{name:<22}{k:<15}{f'{caught}/{n_fraud}':<13}{prec:<13.3f}{cost}")
print("-" * 80)


# ================= PART 3 =================
print("\n" + "=" * 80)
print("PART 3   RANKING AGAINST THRESHOLDING")
print("=" * 80)
print("Same model, same data, same cost function. Only the mechanism differs.\n")

# threshold tuned on out-of-fold training predictions, exactly as before
GRID = np.linspace(0.005, 0.60, 120)
oof = cross_val_predict(hgb(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
costs = []
for t in GRID:
    tn, fp, fn, tp = confusion_matrix(y_tr, (oof >= t).astype(int)).ravel()
    costs.append(FN_COST * fn + FP_COST * fp)
t_cost = GRID[int(np.argmin(costs))]

p = proba["Gradient boosting"]
flags = p >= t_cost
tp = int((flags & (y_va.values == 1)).sum())
n_flag = int(flags.sum())
thr_cost = FN_COST * (n_fraud - tp) + FP_COST * (n_flag - tp)

k_rank, rank_cost, rank_caught = best_budget(cum["Gradient boosting"], 10)

print(f"{'Mechanism':<24}{'Flagged':<12}{'Caught':<13}{'Cost'}")
print("-" * 80)
print(f"{'Cost threshold':<24}{n_flag:<12}{f'{tp}/{n_fraud}':<13}{thr_cost}")
print(f"{'Cost-optimal rank':<24}{k_rank:<12}{f'{rank_caught}/{n_fraud}':<13}{rank_cost}")
print("-" * 80)

d_claims = k_rank - n_flag
d_cost = rank_cost - thr_cost
print(f"\nranking reviews {d_claims:+,} claims and costs {d_cost:+,} relative to the threshold")
print("(negative means ranking is cheaper and opens fewer claims)")
print(f"threshold used: {t_cost:.3f}  (chosen on training data only)")

print("\n" + "=" * 80)
print("Validation has now been scored. It should not be run again.")
print("=" * 80)
