"""
fair_comparison.py

The main experiment. Every approach is given the same chance, and all of them
are judged the way an investigations unit would actually use them.

Three things this fixes about my earlier scripts:

  1. ORDINAL ENCODING
     LabelEncoder sorts category text alphabetically, so "new" was ranking
     ABOVE "more than 7 years", and "none" past claims above "more than 4".
     Nine columns in this dataset are ordered categories. All were scrambled.

  2. A FAIR COMPARISON
     Before, only my approach got a tuned decision threshold and everything
     else was stuck at the default 0.5. Here every approach gets its threshold
     tuned the same way, on training data only.

  3. THE RIGHT METRIC
     Parent paper 4 (Brant & Hobaek Haff 2023) points out that a fraud team
     can only review k cases, so what matters is how many of the top k are
     real fraud. They call the wasted reviews the "fraud loss". That is
     reported here alongside expected cost.

"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
FN_COST = 10        # a missed fraud is paid out in full
FP_COST = 1         # a false alarm wastes an investigator
CAPACITY = 100      # claims the team can review per period

# identifiers and calendar fields: not facts about the claim
ADMIN = ["PolicyNumber", "RepNumber", "Year", "Month", "WeekOfMonth", "DayOfWeek",
         "MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed"]

# ---- the real order of every ordered category ----
ORDER = {
    "AgeOfVehicle": ["new", "2 years", "3 years", "4 years", "5 years",
                     "6 years", "7 years", "more than 7"],
    "VehiclePrice": ["less than 20,000", "20,000 to 29,000", "30,000 to 39,000",
                     "40,000 to 59,000", "60,000 to 69,000", "more than 69,000"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "Days:Policy-Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days:Policy-Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "AddressChange-Claim": ["under 6 months", "1 year", "2 to 3 years",
                            "4 to 8 years", "no change"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    "AgeOfPolicyHolder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
                          "36 to 40", "41 to 50", "51 to 65", "over 65"],
}

# ---------------- load ----------------
df = pd.read_csv("data/carclaims.csv")
df["FraudFound"] = df["FraudFound"].str.strip().map({"Yes": 1, "No": 0})

for col, order in ORDER.items():
    mapping = {v: i for i, v in enumerate(order)}
    assert set(df[col].unique()) <= set(mapping), f"{col}: unmapped value"
    df[col] = df[col].map(mapping)

for col in df.columns:                      # whatever is left is truly nominal
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["FraudFound"] + ADMIN)
y = df["FraudFound"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

n_fraud = int(y_te.sum())
base = y_te.mean()

print(f"{len(df):,} claims, {X.shape[1]} features "
      f"({len(ADMIN)} ID/calendar columns dropped)")
print(f"{len(ORDER)} ordered columns encoded in their real order")
print(f"test set: {len(y_te):,} claims, {n_fraud} fraudulent ({100*base:.1f}%)\n")

GRID = np.linspace(0.005, 0.60, 120)


def cost_at(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return FN_COST * fn + FP_COST * fp


def tune(oof, y_true):
    """lowest-cost threshold, chosen on training predictions only"""
    costs = [cost_at(y_true, (oof >= t).astype(int)) for t in GRID]
    return GRID[int(np.argmin(costs))]


def evaluate(name, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    cost = FN_COST * fn + FP_COST * fp

    ranked = y_te.values[np.argsort(-proba)]
    hits = int(ranked[:CAPACITY].sum())
    fraud_loss = CAPACITY - hits          # wasted reviews, paper 4's metric

    print(f"{name:<32}{thr:<9.3f}{f'{tp}/{n_fraud}':<11}{cost:<9}"
          f"{hits/CAPACITY:<11.3f}{fraud_loss:<13}{hits/CAPACITY/base:.1f}x")
    return proba


def rf(**kw):
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **kw)


def hgb():
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                          learning_rate=0.06)


# ================= the comparison =================
print("=" * 96)
print("EVERY APPROACH GETS ITS THRESHOLD TUNED THE SAME WAY")
print("=" * 96)
print(f"{'Approach':<32}{'Cutoff':<9}{'Caught':<11}{'Cost':<9}"
      f"{'Prec@100':<11}{'Fraud loss':<13}{'Lift'}")
print("-" * 96)

# 1 - random forest, no balancing
oof = cross_val_predict(rf(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
evaluate("Random forest", rf().fit(X_tr, y_tr).predict_proba(X_te)[:, 1],
         tune(oof, y_tr))

# 2 - SMOTE. it goes INSIDE the cross-validation so the threshold is still
#     scored against the real class balance, not the resampled one
pipe = ImbPipeline([("smote", SMOTE(random_state=SEED)), ("clf", rf())])
oof_sm = cross_val_predict(pipe, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
evaluate("SMOTE (parent paper 1)", rf().fit(X_sm, y_sm).predict_proba(X_te)[:, 1],
         tune(oof_sm, y_tr))

# 3 - calibrated random forest
cal = CalibratedClassifierCV(rf(), method="isotonic", cv=5)
oof_c = cross_val_predict(cal, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
evaluate("Random forest + calibration",
         cal.fit(X_tr, y_tr).predict_proba(X_te)[:, 1], tune(oof_c, y_tr))

# 4 - gradient boosting
oof_g = cross_val_predict(hgb(), X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
best = evaluate("Gradient boosting",
                hgb().fit(X_tr, y_tr).predict_proba(X_te)[:, 1], tune(oof_g, y_tr))

# 5 - gradient boosting, calibrated
cal_g = CalibratedClassifierCV(hgb(), method="isotonic", cv=5)
oof_cg = cross_val_predict(cal_g, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
evaluate("Gradient boosting + calibration",
         cal_g.fit(X_tr, y_tr).predict_proba(X_te)[:, 1], tune(oof_cg, y_tr))

print("-" * 96)
print(f"\nFraud loss = wasted reviews out of {CAPACITY}. Lower is better.")
print(f"Reviewing {CAPACITY} claims at random would give precision {base:.3f}.\n")


# ================= capacity curve for the best model =================
print("=" * 96)
print("CAPACITY CURVE  (gradient boosting)")
print("=" * 96)
print("What the team gets for each weekly review budget.\n")
print(f"{'Capacity K':<13}{'Cutoff':<10}{'Caught':<12}{'Precision':<12}"
      f"{'Fraud loss':<13}{'% of fraud':<12}{'Lift'}")
print("-" * 96)

order = np.argsort(-best)
ranked, scores = y_te.values[order], best[order]

for k in [50, 100, 200, 300, 500, 1000]:
    hits = int(ranked[:k].sum())
    print(f"{k:<13}{scores[k-1]:<10.3f}{f'{hits}/{n_fraud}':<12}{hits/k:<12.3f}"
          f"{k-hits:<13}{100*hits/n_fraud:<12.1f}{hits/k/base:.1f}x")

print("-" * 96)
print("\nSmaller budgets give better precision but find less of the total fraud.")
print("That trade is the thing none of parent papers 1, 2 or 3 reports.")
