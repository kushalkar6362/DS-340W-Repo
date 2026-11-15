"""
split_data.py

Splits carclaims.csv into training / test / validation at 70 / 20 / 10.

The validation set is written once and then left alone. Nothing in
novel_approach.py reads it. It gets scored at the very end, one time.

Two modes:

  --mode clean   Split the raw data first. Test and validation keep the real
                 ~6% fraud rate. This is the honest setup and the default.

  --mode paper   Balance the whole dataset with SMOTE and then split, which is
                 what parent paper 1 does. Kept so the leakage demo still works.
                 Synthetic claims built from training rows leak into test.

Both modes now use the corrected ordinal encoding and drop the nine
identifier and calendar columns.

Usage:
    python split_data.py --input data/carclaims.csv --mode clean
    python split_data.py --input data/carclaims.csv --mode paper

Writes to ./splits/<mode>/train.csv, test.csv, validation.csv
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

SEED = 42
TARGET = "FraudFound"

# identifiers and calendar fields, not facts about the claim
ADMIN = ["PolicyNumber", "RepNumber", "Year", "Month", "WeekOfMonth", "DayOfWeek",
         "MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed"]

# the real order of every ordered category
# LabelEncoder sorts alphabetically, which put "new" above "more than 7"
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


def load_and_encode(path):
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].str.strip().map({"Yes": 1, "No": 0})

    # ordered categories first, in their real order
    for col, order in ORDER.items():
        mapping = {v: i for i, v in enumerate(order)}
        assert set(df[col].unique()) <= set(mapping), f"{col}: unmapped value"
        df[col] = df[col].map(mapping)

    # whatever is left and still text is genuinely nominal
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col])

    df = df.drop(columns=ADMIN)
    return df


def write(outdir, name, X, y):
    frame = X.copy()
    frame[TARGET] = y.values if hasattr(y, "values") else y
    path = os.path.join(outdir, f"{name}.csv")
    frame.to_csv(path, index=False)
    n = int(frame[TARGET].sum())
    print(f"  {name:<11}{len(frame):>7,} rows    {n:>5,} fraud  ({100*n/len(frame):5.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/carclaims.csv")
    ap.add_argument("--mode", choices=["clean", "paper"], default="clean")
    ap.add_argument("--outdir", default="splits")
    args = ap.parse_args()

    df = load_and_encode(args.input)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"Loaded {len(df):,} claims, {X.shape[1]} features "
          f"({len(ADMIN)} ID/calendar columns dropped)")
    print(f"{len(ORDER)} ordered columns encoded in their real order")
    print(f"Fraud rate: {100*y.mean():.1f}%\n")

    # 70 / 20 / 10, stratified so every split keeps the same fraud rate
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_te, X_val, y_te, y_val = train_test_split(
        X_hold, y_hold, test_size=(1 / 3), stratify=y_hold, random_state=SEED)

    if args.mode == "paper":
        # balance everything then re-split, mirroring parent paper 1
        X_bal, y_bal = SMOTE(random_state=SEED).fit_resample(X, y)
        X_tr, X_hold, y_tr, y_hold = train_test_split(
            X_bal, y_bal, test_size=0.30, stratify=y_bal, random_state=SEED)
        X_te, X_val, y_te, y_val = train_test_split(
            X_hold, y_hold, test_size=(1 / 3), stratify=y_hold, random_state=SEED)
        print("PAPER MODE: SMOTE applied to the full dataset before splitting.")
        print("Synthetic claims built from training rows are now in test.\n")

    outdir = os.path.join(args.outdir, args.mode)
    os.makedirs(outdir, exist_ok=True)

    print(f"Writing to {outdir}/")
    write(outdir, "train", X_tr, y_tr)
    write(outdir, "test", X_te, y_te)
    write(outdir, "validation", X_val, y_val)

    print("\nValidation is not read by any other script.")
    print("Score it once, at the end, after training and test results are final.")


if __name__ == "__main__":
    main()
