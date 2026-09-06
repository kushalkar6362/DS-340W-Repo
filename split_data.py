"""
Split carclaims.csv into training / test / validation sets (70 / 20 / 10).

Two modes:

  --mode paper   Balance the FULL dataset with SMOTE, then split.
                 This is what the parent paper does. Reproduces its numbers.
                 Note: synthetic points derived from training rows can land in
                 the test set, which inflates reported metrics.

  --mode clean   Split the RAW data first, then apply SMOTE to the training
                 set only. Test and validation keep the true ~6% fraud rate.
                 This is the honest evaluation.

Usage:
    python split_data.py --input carclaims.csv --mode paper
    python split_data.py --input carclaims.csv --mode clean

Outputs land in ./splits/<mode>/ as train.csv, test.csv, validation.csv.
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

SEED = 42
TARGET_CANDIDATES = ["FraudFound_P", "FraudFound", "fraud_reported", "FraudFound_p"]


def find_target(df):
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise SystemExit(
        f"Could not find the target column. Looked for {TARGET_CANDIDATES}.\n"
        f"Columns present: {list(df.columns)}"
    )


def encode(df, target):
    """Label-encode every categorical column. Map the target to 0/1."""
    df = df.copy()

    if df[target].dtype == object:
        df[target] = (
            df[target].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0})
        )
        if df[target].isna().any():
            raise SystemExit(f"Could not map target values in '{target}' to 0/1.")
    df[target] = df[target].astype(int)

    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def standardize(X_fit, *others):
    """Fit the scaler on X_fit only, then apply it everywhere."""
    scaler = StandardScaler().fit(X_fit)
    out = [pd.DataFrame(scaler.transform(X_fit), columns=X_fit.columns, index=X_fit.index)]
    for X in others:
        out.append(pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index))
    return out


def write(outdir, name, X, y, target):
    frame = X.copy()
    frame[target] = y.values if hasattr(y, "values") else y
    path = os.path.join(outdir, f"{name}.csv")
    frame.to_csv(path, index=False)
    n_fraud = int(frame[target].sum())
    pct = 100.0 * n_fraud / len(frame)
    print(f"  {name:<11} {len(frame):>7,} rows   fraud: {n_fraud:>6,} ({pct:5.2f}%)   -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="carclaims.csv")
    ap.add_argument("--mode", choices=["paper", "clean"], default="paper")
    ap.add_argument("--outdir", default="splits")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    target = find_target(df)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns. Target column: {target}")
    print(f"Raw class balance: {int(df[target].astype(str).str.strip().str.lower().isin(['yes','1']).sum()):,} fraud")

    df = encode(df, target)
    X = df.drop(columns=[target])
    y = df[target]

    outdir = os.path.join(args.outdir, args.mode)
    os.makedirs(outdir, exist_ok=True)

    if args.mode == "paper":
        # Balance everything first, then split. Mirrors the parent paper.
        X_bal, y_bal = SMOTE(random_state=SEED).fit_resample(X, y)
        print(f"\nSMOTE applied to full dataset: {len(X_bal):,} rows, balanced.")

        X_train, X_hold, y_train, y_hold = train_test_split(
            X_bal, y_bal, test_size=0.30, stratify=y_bal, random_state=SEED)
        X_test, X_val, y_test, y_val = train_test_split(
            X_hold, y_hold, test_size=(1 / 3), stratify=y_hold, random_state=SEED)

    else:
        # Split raw data first. SMOTE the training set only.
        X_train, X_hold, y_train, y_hold = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEED)
        X_test, X_val, y_test, y_val = train_test_split(
            X_hold, y_hold, test_size=(1 / 3), stratify=y_hold, random_state=SEED)

        X_train, y_train = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
        print(f"\nSMOTE applied to training set only: {len(X_train):,} rows, balanced.")
        print("Test and validation retain the original class ratio.")

    X_train, X_test, X_val = standardize(X_train, X_test, X_val)

    print(f"\nWriting splits ({args.mode} mode):")
    write(outdir, "train", X_train, y_train, target)
    write(outdir, "test", X_test, y_test, target)
    write(outdir, "validation", X_val, y_val, target)

    print("\nValidation set is not to be scored until training and test results are final.")


if __name__ == "__main__":
    main()
