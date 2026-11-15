"""
marginal_value.py

This flips the question around.

Parent paper 2 fixes a cost ratio and finds a threshold.
Parent paper 4 takes the review budget as given and works inside it.
Neither one asks what the budget should actually be.

But the cost function already answers that. Reviewing one more claim costs 1.
Catching one more fraud saves 10. So it is worth reviewing the next claim on
the list for as long as the chance it is fraudulent stays above 1/10. Once the
ranking gets thin enough that the next claim is fraudulent less than one time
in ten, the review stops paying for itself.

That gives an economically optimal budget instead of an assumed one.

PART 1  the optimal budget for each model at a 10-to-1 ratio
PART 2  how that budget moves when the cost ratio changes

Part 2 also deals with the comment three reviewers made at the midterm, that
the 10-to-1 ratio was assumed. It is still assumed, but now you can see
exactly what it changes.

Validation is never read.

Run:  python marginal_value.py
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

SEED = 42
FP_COST = 1
TARGET = "FraudFound"


def load(name):
    df = pd.read_csv(f"splits/clean/{name}.csv")
    return df.drop(columns=[TARGET]), df[TARGET]


X_tr, y_tr = load("train")
X_te, y_te = load("test")

n_fraud = int(y_te.sum())
n_test = len(y_te)
base = y_te.mean()

print(f"test set: {n_test:,} claims, {n_fraud} fraud ({100*base:.1f}%)\n")


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

print("fitting models...\n")
cum = {}
for name, m in MODELS.items():
    p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    ranked = y_te.values[np.argsort(-p)]
    cum[name] = np.concatenate([[0], np.cumsum(ranked)])   # hits after reviewing k


def best_budget(hits, fn_cost):
    """
    cost(k) = fn_cost * frauds missed + 1 * false alarms
    walk every possible k and take the cheapest
    """
    k = np.arange(len(hits))
    cost = fn_cost * (n_fraud - hits) + FP_COST * (k - hits)
    best = int(np.argmin(cost))
    return best, int(cost[best]), int(hits[best])


# ============ PART 1 ============
print("=" * 82)
print("PART 1   ECONOMICALLY OPTIMAL REVIEW BUDGET  (10-to-1 cost ratio)")
print("=" * 82)
print(f"{'Model':<22}{'Best budget':<14}{'Caught':<12}{'Precision':<12}"
      f"{'Recall':<10}{'Cost'}")
print("-" * 82)

for name, hits in cum.items():
    k, cost, caught = best_budget(hits, 10)
    prec = caught / k if k else 0
    print(f"{name:<22}{k:<14}{f'{caught}/{n_fraud}':<12}{prec:<12.3f}"
          f"{caught/n_fraud:<10.3f}{cost}")

print("-" * 82)
print("\nThis is the point where the next claim on the list is fraudulent")
print("less than one time in ten, so reviewing it stops paying for itself.")


# ============ PART 2 ============
print("\n" + "=" * 82)
print("PART 2   HOW THE OPTIMAL BUDGET MOVES WITH THE COST RATIO")
print("=" * 82)
print("Three reviewers asked why the ratio is 10-to-1. Here is what it changes.\n")

RATIOS = [2, 5, 10, 15, 20, 30]
print(f"{'Model':<22}" + "".join(f"{str(r)+'-to-1':<11}" for r in RATIOS))
print("-" * 82)
for name, hits in cum.items():
    row = ""
    for r in RATIOS:
        k, _, _ = best_budget(hits, r)
        row += f"{k:<11}"
    print(f"{name:<22}{row}")
print("-" * 82)

print("\nA carrier that treats a missed fraud as only twice the cost of a wasted")
print("review should staff a much smaller unit than one that treats it as 30x.")
print("The ratio does not change which model to use. It changes how many people")
print("to put on the queue.")
