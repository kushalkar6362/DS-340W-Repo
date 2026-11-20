# Automobile Insurance Claim Fraud Detection Under a Review Budget

Kushal Kar · Group 70 · DS 340W · The Pennsylvania State University

Fraud models in the automobile insurance literature are tuned for accuracy, or for
cost in the abstract. Neither matches how an insurer works. A special investigations
unit can only open a fixed number of claims a week, so what matters is how much real
fraud sits at the top of the ranking, not how the model scores across every claim at
once.

This repository reproduces four published approaches, corrects two faults found along
the way, and evaluates everything under that constraint.

---

## Quick start

```bash
git clone https://github.com/kushalkar6362/DS-340W-Repo.git
cd DS-340W-Repo

python -m venv .venv
.venv\Scripts\activate              # Windows PowerShell
# source .venv/bin/activate         # macOS or Linux

pip install -r requirements.txt

python split_data.py --input data/carclaims.csv --mode clean
python leakage_test.py
```

`leakage_test.py` finishes in under a minute and reproduces the headline result from
parent paper 1. Everything else is below.

**Python 3.11 is what this was developed and tested on.** The pinned versions in
`requirements.txt` matter, in particular `pandas==2.2.3`, because pandas 3.0 changes
how string columns are typed and the encoding step behaves differently.

---

## Running everything, with timings

Run `split_data.py` first. Every other script reads from `splits/clean/`.

| Script | Runtime | What it produces |
|---|---|---|
| `split_data.py` | seconds | the 70 / 20 / 10 partitions |
| `leakage_test.py` | under 1 min | parent paper 1 reproduction and the leakage result |
| `cost_sensitive.py` | 2 to 3 min | first cost comparison, superseded |
| `capacity_and_ratio.py` | 3 to 5 min | cost ratio sweep and the PolicyNumber finding |
| `capacity_threshold.py` | 5 min | clean feature re-run, superseded |
| `fair_comparison.py` | 5 to 10 min | five approaches with equally tuned thresholds |
| `novel_approach.py` | 15 to 20 min | the five selection rules, model by budget grid |
| `marginal_value.py` | 10 min | cost optimal review budget |
| `score_validation.py` | 8 to 12 min | the held-out validation set, scored once |

Saved output from every script is in `results/`, so the numbers can be checked
without waiting for a full run.

**Reproducibility.** Everything is seeded at 42. Every result reproduces exactly on
re-run except the SMOTE rows, which can move by one or two claims across machines.
SMOTE's nearest neighbour search produces tied distances on label encoded features,
and those ties break according to the underlying numerical library. The direction and
magnitude of every conclusion is unaffected.

---

## Results

### The parent paper reproduces, and so does its flaw

`leakage_test.py` trains the same model twice on the same data with the same seed,
changing only the order of two steps.

| Step order | Accuracy | Recall |
|---|---|---|
| Balance then split (parent paper 1) | 0.9569 | 0.9583 |
| Split then balance (corrected) | 0.9125 | 0.1514 |

Accuracy barely moves. Recall collapses. Balancing before partitioning places
synthetic fraud cases derived from training rows into the test set, so the model is
evaluated on data it has effectively already seen.

### Giving every approach the same tuned threshold

`fair_comparison.py`, test set of 3,084 claims containing 185 frauds.

| Approach | Cutoff | Cost | Precision @100 | Lift |
|---|---|---|---|---|
| SMOTE (parent paper 1) | 0.095 | 1,458 | 0.150 | 2.5x |
| GB + calibration | 0.085 | 1,343 | 0.190 | 3.2x |
| Random forest | 0.060 | 1,434 | 0.250 | 4.2x |
| RF + calibration | 0.090 | 1,439 | 0.270 | 4.5x |
| **Gradient boosting** | **0.045** | **1,313** | **0.290** | **4.8x** |

Once every approach gets the same threshold tuning, SMOTE is the worst of the five and
performs below using no balancing at all. The synthetic claims degrade the probability
ranking, and the ranking is what matters when only the top claims are reviewed. All
three of the original parent papers resample.

### Selection rules compared

`novel_approach.py`, budget K = 100.

| Rule | Source | Flagged | Caught | Cost | Within budget |
|---|---|---|---|---|---|
| Fixed 0.5 | paper 3 | 8 | 6 / 185 | 1,792 | yes |
| Cost threshold | paper 2 | 1,431 | 177 / 185 | 1,334 | no |
| Top-K ranking | paper 4 | 100 | 26 / 185 | 1,664 | yes |
| Budgeted cost | proposed | 35 | 15 / 185 | 1,720 | yes |
| Cost weighted | control | 22 | 10 / 185 | 1,762 | yes |

The proposed budgeted threshold used only 35 of its 100 available reviews. Out of fold
probabilities come from models trained on four fifths of the data while deployment
probabilities come from a model trained on all of it, so a threshold calibrated to flag
K flags considerably fewer. Ranking is invariant to that shift and thresholding is not.

### No single model wins at every budget

| Model | K=50 | K=100 | K=200 | K=300 | K=500 | K=1000 |
|---|---|---|---|---|---|---|
| Random forest | **0.360** | **0.270** | 0.220 | 0.200 | 0.170 | 0.133 |
| RF + calibration | 0.340 | 0.260 | 0.220 | 0.187 | 0.166 | 0.131 |
| Gradient boosting | 0.340 | 0.260 | **0.245** | **0.213** | **0.172** | 0.139 |
| GB + calibration | 0.360 | 0.260 | 0.180 | 0.160 | 0.160 | **0.143** |

Random forest is best at small budgets, gradient boosting in the middle, calibrated
gradient boosting at 1,000. A carrier cannot choose a model before deciding its
review capacity.

### Deriving the budget instead of assuming it

`marginal_value.py`. Reviewing one more claim costs 1 and catching one more fraud
saves 10, so the next claim on the list is worth opening while its fraud probability
exceeds one tenth.

| Model | 2:1 | 5:1 | 10:1 | 15:1 | 20:1 | 30:1 | Cost at 10:1 |
|---|---|---|---|---|---|---|---|
| Random forest | 38 | 210 | 1,126 | 1,290 | 1,406 | 1,564 | 1,337 |
| RF + calibration | 20 | 153 | 741 | 1,255 | 1,454 | 1,574 | 1,315 |
| Gradient boosting | 6 | 260 | 1,254 | 1,393 | 1,441 | 1,530 | 1,256 |
| GB + calibration | 29 | 34 | 1,046 | 1,321 | 1,394 | 1,394 | **1,246** |

The budget of 100 assumed throughout the earlier work is roughly an order of magnitude
too small. The assumed cost ratio changes the recommended staffing level by two orders
of magnitude without changing which model performs best.

**Ranking dominates thresholding on the threshold's own measure.** Holding the model,
the test set and the cost function fixed, the cost minimising threshold flags 1,431
claims and costs 1,334, while selecting the cost minimising number by rank flags 1,254
and costs 1,256. Ranking reviews 177 fewer claims and still costs 78 less.

---

## Two corrections to my own work

Both are kept in the commit history deliberately.

**The model was reading claim ID numbers.** Feature importances showed `PolicyNumber`
as the most important feature. It is an identifier and carries no information about
fraud. Dropping nine identifier and calendar columns cut the reported improvement from
35 percent to 10.5 percent. Nothing in the accuracy or cost figures would have revealed
this. Only the explainability check did.

**The categorical encoding was scrambled.** `LabelEncoder` sorts category text
alphabetically, so `"new"` ranked above `"more than 7"` for vehicle age, and `"none"`
above `"more than 4"` for past claims. Nine ordered columns were affected and are now
encoded in their real order.

---

## Repository layout

```
DS-340W-Repo/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── carclaims.csv              15,420 claims, 33 fields
│
├── splits/
│   ├── clean/                     train / test / validation, split before balancing
│   └── paper/                     parent paper 1 ordering, for the leakage demo
│
├── notebooks/
│   └── vehicle-insurance-fraud-detection.ipynb    baseline, UNMODIFIED
│
├── split_data.py                  70 / 20 / 10 stratified split, seed 42
├── leakage_test.py                reproduces parent paper 1 and its leakage
├── cost_sensitive.py              first cost comparison
├── capacity_and_ratio.py          cost ratio sweep, feature importances
├── capacity_threshold.py          clean feature re-run
├── fair_comparison.py             equally tuned thresholds across five approaches
├── novel_approach.py              five selection rules, model by budget grid
├── marginal_value.py              cost optimal review budget
├── score_validation.py            the only script that reads the validation split
│
└── results/                       saved output from every script
```

Earlier scripts are kept rather than deleted. `leakage_test.py` is the evidence behind
the parent paper 1 critique, and the others document how the results changed as the
two faults above were found.

---

## Data splits

70 / 20 / 10, stratified on the fraud label, seed fixed at 42.

| Split | Rows | Fraud | Purpose |
|---|---|---|---|
| Training | 10,794 | 646 (5.98%) | model fitting, threshold selection, five fold CV |
| Test | 3,084 | 185 (6.00%) | every result reported above |
| Validation | 1,542 | 92 (5.97%) | held out, scored once at the end |

No training or tuning code reads the validation partition. `score_validation.py` is the
only script that opens it, it was run once after every other result was final, and its
saved output is in `results/`. Budgets on that partition are halved so the review rate
matches the test partition, so a budget of 100 on test corresponds to 50 there.

`--mode clean` splits first so test and validation keep the true fraud rate.
`--mode paper` balances the full dataset before splitting, reproducing parent paper 1's
pipeline so the leakage result can be demonstrated.

---

## Parent papers

**1.** Özaltın, Ö. and Karadağ Erdemir, Ö. (2025). *Detecting automobile insurance fraud
using a novel penalty-driven feature selection method with particle swarm optimization
and machine learning classifiers.* Scientific Reports, 15, art. 41793.
<https://www.nature.com/articles/s41598-025-25700-2>

Source of the dataset. Applies SMOTE before the train and test split.

**2.** Bekkaye, C., Zari, T. and Guerbaz, R. (2026). *Reliable auto insurance fraud
detection using boosting and deep learning models.* Discover Artificial Intelligence.

Source of the expected cost function, 10 × missed frauds + 1 × false alarms.

**3.** Olivia, D., Khan, Z. and Shetty, S. (2025). *A machine learning and explainable
artificial intelligence approach for insurance fraud classification.* Inteligencia
Artificial, 28(75), 140–169.
<https://journal.iberamia.org/index.php/intartif/article/view/1656>

Same dataset. Source of the explainability check that caught the `PolicyNumber` fault.

**4.** Brant, S. B. and Hobæk Haff, I. (2023). *The fraud loss for selecting the model
complexity in fraud detection.* Journal of Applied Statistics, 50(10), 2209–2227.
<https://arxiv.org/abs/2101.11907>

Defines the fraud loss, the count of non-fraudulent cases among the top k reviewed.

---

## Dataset

Angoss KnowledgeSeeker automobile insurance dataset, originally released by EXL
Service. Claims from January 1994 through December 1996.

| | |
|---|---|
| Records | 15,420 |
| Features | 33 (23 after dropping identifiers and calendar fields) |
| Legitimate | 14,497 (94%) |
| Fraudulent | 923 (6%) |

Hosted in this repository at `data/carclaims.csv`. Also available at
<https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection>

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'imblearn'`** — the package is
`imbalanced-learn` and installs from `requirements.txt`. Confirm the virtual
environment is active, the prompt should start with `(.venv)`.

**`FileNotFoundError: splits/clean/train.csv`** — run `split_data.py` first.

**`TypeError: Invalid value for dtype`** — you are on pandas 3.x. Install the pinned
version with `pip install pandas==2.2.3`.

**PowerShell blocks activation** — run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.

**A script appears to hang** — `novel_approach.py` cross validates four models
including two calibrated ones and takes 15 to 20 minutes with no output in between.
This is expected.
