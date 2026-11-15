"""
novel_approach.py

Week 11. The modular block I am replacing is the SELECTION RULE, meaning how
the model goes from a predicted probability to a decision about which claims
an investigator actually opens.

Everything before that block stays the same: same split, same encoding, same
models. Only the selection rule changes, so any difference is down to that.

Five selection rules compared:

  1. FIXED 0.5          flag anything above 0.5          (parent paper 3)
  2. COST THRESHOLD     cutoff that minimises 10*FN + FP (parent paper 2)
  3. TOP K              rank and take the top K, no cutoff at all (parent paper 4)
  4. BUDGETED COST      minimise cost but never flag more than K  <-- MY APPROACH
  5. COST WEIGHTED      handle cost during training instead, then use 0.5

Rule 4 is the new one. Papers 2 and 4 each solve half of this. Paper 2 picks a
cutoff on money and ignores whether anyone can work the resulting list. Paper 4
caps the list at K and ignores what the errors cost. Rule 4 does both.

Then two more tables:

  PART 2  does gradient boosting still win at every budget, or only at 100?
          reviewers asked this and I did not know, so this answers it
  PART 3  can the budget metric pick the model on training data alone,
          without looking at test? this is paper 4's idea applied to models

The validation split is never read by this script.

Run:  python novel_approach.py
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

SEED = 42
FN_COST = 10       # a missed fraud is paid out in full
FP_COST = 1        # a false alarm wastes an investigator
BUDGET = 100       # claims the team can review per period
TARGET = "FraudFound"

GRID = np.linspace(0.005, 0.60, 120)


# ---------------------------------------------------------------- data
def load(name):
    df = pd.read_csv(f"splits/clean/{name}.csv")
    return df.drop(columns=[TARGET]), df[TARGET]


X_tr, y_tr = load("train")
X_te, y_te = load("test")
# validation deliberately not loaded

n_fraud = int(y_te.sum())
base = y_te.mean()

print(f"train {len(y_tr):,} claims   test {len(y_te):,} claims, {n_fraud} fraud "
      f"({100*base:.1f}%)")
print(f"review budget K = {BUDGET}\n")


# ---------------------------------------------------------------- models
def rf(**kw):
    return RandomForestClassifier(n_estimators=300, random_state=SEED,
                                  n_jobs=-1, **kw)


def hgb():
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                          learning_rate=0.06)


def cost_of(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return FN_COST * fn + FP_COST * fp


def score(name, flags):
    """flags is a boolean array over the test set"""
    tp = int((flags & (y_te.values == 1)).sum())
    n_flag = int(flags.sum())
    fn = n_fraud - tp
    fp = n_flag - tp
    cost = FN_COST * fn + FP_COST * fp
    prec = tp / n_flag if n_flag else 0.0
    print(f"{name:<22}{n_flag:<10}{f'{tp}/{n_fraud}':<11}{prec:<12.3f}"
          f"{cost:<9}{'yes' if n_flag <= BUDGET else 'NO':<10}")


# out-of-fold training predictions, used to pick every cutoff
# the test set never decides anything
print("fitting models and getting out-of-fold training predictions...\n")
oof = cross_val_predict(hgb(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
model = hgb().fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]


# ================= PART 1 — the five selection rules =================
print("=" * 78)
print("PART 1   FIVE SELECTION RULES, SAME MODEL AND SAME DATA")
print("=" * 78)
print(f"{'Rule':<22}{'Flagged':<10}{'Caught':<11}{'Precision':<12}"
      f"{'Cost':<9}{'Fits budget'}")
print("-" * 78)

# 1 - fixed 0.5
score("1. Fixed 0.5", proba >= 0.5)

# 2 - cost minimising cutoff, no cap on how many get flagged
costs = [cost_of(y_tr, (oof >= t).astype(int)) for t in GRID]
t_cost = GRID[int(np.argmin(costs))]
score("2. Cost threshold", proba >= t_cost)

# 3 - pure ranking, take the top K
order = np.argsort(-proba)
top_k = np.zeros(len(proba), dtype=bool)
top_k[order[:BUDGET]] = True
score("3. Top K", top_k)

# 4 - MY RULE: lowest cost among the cutoffs that flag at most K
#     the cap is checked on training data, same as the cutoff
feasible = []
for t in GRID:
    n_flag_tr = int((oof >= t).sum())
    # scale the training count down to test size so the cap means the same thing
    if n_flag_tr * len(y_te) / len(y_tr) <= BUDGET:
        feasible.append((cost_of(y_tr, (oof >= t).astype(int)), t))
t_budget = min(feasible)[1] if feasible else GRID[-1]
score("4. Budgeted cost", proba >= t_budget)

# 5 - cost handled during training instead of after
w = rf(class_weight={0: FP_COST, 1: FN_COST}).fit(X_tr, y_tr)
score("5. Cost weighted", w.predict_proba(X_te)[:, 1] >= 0.5)

print("-" * 78)
print(f"\ncutoffs: cost threshold {t_cost:.3f}   budgeted cost {t_budget:.3f}")
print(f"reviewing {BUDGET} claims at random would give precision {base:.3f}")


# ================= PART 2 — does the winner change with budget? =================
print("\n" + "=" * 78)
print("PART 2   PRECISION BY MODEL AND BUDGET")
print("=" * 78)
print("Reviewers asked whether gradient boosting wins at every budget.\n")

MODELS = {
    "Random forest": rf(),
    "RF + calibration": CalibratedClassifierCV(rf(), method="isotonic", cv=5),
    "Gradient boosting": hgb(),
    "GB + calibration": CalibratedClassifierCV(hgb(), method="isotonic", cv=5),
}
BUDGETS = [k for k in [50, 100, 200, 300, 500, 1000] if k <= len(y_te)]

probs = {}
for name, m in MODELS.items():
    probs[name] = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

print(f"{'Model':<22}" + "".join(f"K={k:<9}" for k in BUDGETS))
print("-" * 78)
for name, p in probs.items():
    ranked = y_te.values[np.argsort(-p)]
    row = "".join(f"{int(ranked[:k].sum())/k:<11.3f}" for k in BUDGETS)
    print(f"{name:<22}{row}")
print("-" * 78)

best = {}
for k in BUDGETS:
    scores = {n: int(y_te.values[np.argsort(-p)][:k].sum()) / k for n, p in probs.items()}
    best[k] = max(scores, key=scores.get)
print("\nbest model at each budget:")
for k in BUDGETS:
    print(f"  K={k:<6} {best[k]}")


# ================= PART 3 — pick the model without touching test =================
print("\n" + "=" * 78)
print("PART 3   CAN THE BUDGET METRIC PICK THE MODEL ON TRAINING DATA ALONE?")
print("=" * 78)
print("Paper 4 uses fraud loss to choose model complexity. Here I use it to\n"
      "choose the model itself, scoring only on out-of-fold training folds,\n"
      "then check whether that choice matches what actually wins on test.\n")

print(f"{'Model':<22}{'train prec@K':<16}{'test prec@K':<16}{'fraud loss (test)'}")
print("-" * 78)

train_scores, test_scores = {}, {}
for name, m in MODELS.items():
    o = cross_val_predict(m, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
    # scale the budget to the training set so the comparison is fair
    k_tr = int(round(BUDGET * len(y_tr) / len(y_te)))
    tr_hits = int(y_tr.values[np.argsort(-o)][:k_tr].sum())
    te_hits = int(y_te.values[np.argsort(-probs[name])][:BUDGET].sum())
    train_scores[name] = tr_hits / k_tr
    test_scores[name] = te_hits / BUDGET
    print(f"{name:<22}{tr_hits/k_tr:<16.3f}{te_hits/BUDGET:<16.3f}{BUDGET - te_hits}")

print("-" * 78)
pick = max(train_scores, key=train_scores.get)
actual = max(test_scores, key=test_scores.get)
print(f"\npicked on training data:  {pick}")
print(f"actually best on test:    {actual}")
print("MATCH" if pick == actual else "MISMATCH, the training signal did not transfer")
