# Automobile Insurance Claim Fraud Detection

Kushal Kar · Group 70 · DS 340W · Pennsylvania State University

Fraud models in this literature are tuned for accuracy, or for cost in the abstract.
Neither matches how an insurer actually works: a special investigations unit can only
open a fixed number of claims a week, so what matters is how much real fraud sits at
the top of the ranking, not how the model scores across every claim at once.

This project re-runs four published approaches under that constraint.

---

## Headline results

From `experiments/fair_comparison.py`, on a held-out test set of 3,084 claims
containing 185 frauds (6.0%). Every approach gets its decision threshold tuned the
same way, on training data only.

| Approach | Cutoff | Expected cost | Precision @100 | Wasted reviews | Lift |
|---|---|---|---|---|---|
| SMOTE (parent paper 1) | 0.095 | 1,458 | 0.150 | 85 | 2.5x |
| Gradient boosting + calibration | 0.085 | 1,343 | 0.190 | 81 | 3.2x |
| Random forest | 0.060 | 1,434 | 0.250 | 75 | 4.2x |
| Random forest + calibration | 0.090 | 1,439 | 0.270 | 73 | 4.5x |
| **Gradient boosting** | **0.045** | **1,313** | **0.290** | **71** | **4.8x** |

**Once every approach gets the same threshold tuning, SMOTE is the worst of the five
— worse than doing no balancing at all.** The synthetic claims degrade the probability
ranking, and ranking is what matters when only the top claims get reviewed. All three
of the original parent papers resample.

### What a review budget actually buys

| Capacity | Cutoff | Fraud found | Precision | Wasted reviews | Lift |
|---|---|---|---|---|---|
| 50 | 0.267 | 19 / 185 | 0.380 | 31 | 6.3x |
| 100 | 0.215 | 29 / 185 | 0.290 | 71 | 4.8x |
| 200 | 0.179 | 42 / 185 | 0.210 | 158 | 3.5x |
| 300 | 0.159 | 57 / 185 | 0.190 | 243 | 3.2x |
| 500 | 0.125 | 86 / 185 | 0.172 | 414 | 2.9x |
| 1,000 | 0.084 | 138 / 185 | 0.138 | 862 | 2.3x |

A smaller budget is more accurate per review but finds less of the total fraud. No
setting wins both, which is why the budget has to be stated before a model is chosen.

---

## Two corrections to my own work

Both are kept in the history on purpose.

**1. The model was reading claim ID numbers.** Feature importances showed
`PolicyNumber` as the single most important feature. It is an identifier and carries
no information about fraud. Dropping nine ID and calendar columns cut the reported
improvement from 35% to 10.5%. Nothing in the accuracy or cost figures would have
revealed this — only the explainability check did.

**2. The categorical encoding was scrambled.** `LabelEncoder` sorts category text
alphabetically, so `"new"` ranked above `"more than 7"` for vehicle age, and `"none"`
above `"more than 4"` for past claims. Nine ordered columns were affected.
`fair_comparison.py` encodes them in their real order.

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
├── notebooks/
│   ├── carclaims.csv              copy the notebook reads directly
│   └── vehicle-insurance-fraud-detection.ipynb
│                                  parent paper 1's baseline, UNMODIFIED
│
├── split_data.py                  70 / 20 / 10 stratified split, seed 42
├── leakage_test.py                week 5 — reproduces paper 1's leakage
├── cost_sensitive.py              week 6 — four approaches under a cost function
├── capacity_and_ratio.py          week 7 — cost ratio sweep, first capacity look
├── capacity_threshold.py          week 8 — re-run after dropping ID columns
└── fair_comparison.py             week 8 — MAIN EXPERIMENT, supersedes the above
```

Every script is run from the repository root and reads `data/carclaims.csv`.

Earlier experiment scripts are kept rather than deleted. `leakage_test.py` is the
evidence behind the parent paper 1 critique, and the rest document how the results
changed as the two bugs above were found.

---

## Running it

```bash
git clone https://github.com/kushalkar6362/DS-340W-Repo.git
cd DS-340W-Repo

python -m venv .venv
.venv\Scripts\activate             # Windows
# source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt

python fair_comparison.py
```

Runs in roughly 5–10 minutes. Everything except the SMOTE row is deterministic at
`SEED = 42`. The SMOTE row can shift by a few claims across machines, because its
nearest-neighbour search produces tied distances on label-encoded features and those
ties break differently depending on the underlying numerical library. The direction
and size of the result are unaffected.

### Generating the splits

70 / 20 / 10, stratified on the fraud label, seed fixed at 42.

```bash
python split_data.py --input data/carclaims.csv --mode paper
python split_data.py --input data/carclaims.csv --mode clean
```

| Split | Share | Purpose |
|---|---|---|
| Training | 70% | Model fitting, threshold selection, five-fold CV |
| Test | 20% | Comparison against the papers' reported tables |
| Validation | 10% | Held out. Scored exactly once, at the end. |

The validation set is not read by any training or tuning code.

`--mode paper` balances first and then splits, reproducing parent paper 1's published
pipeline so the numbers are comparable. `--mode clean` splits first and applies SMOTE
to the training set only, so test and validation keep the true ~6% fraud rate.

---

## Parent papers

**1. Özaltın, Ö. & Karadağ Erdemir, Ö. (2025).** *Detecting automobile insurance fraud
using a novel penalty-driven feature selection method with particle swarm optimization
and machine learning classifiers.* **Scientific Reports**, 15, 41793.
<https://www.nature.com/articles/s41598-025-25700-2> · DOI 10.1038/s41598-025-25700-2 · CC BY

Source of the dataset. Applies SMOTE to the full dataset before splitting, so synthetic
points interpolated from training rows appear in the test set. Reproduced in
`leakage_test.py`: correcting the step order drops recall from 0.96 to 0.14.

**2. Bekkaye, C., Zari, T. & Guerbaz, R. (2026).** *Reliable auto insurance fraud
detection using boosting and deep learning models.* **Discover Artificial Intelligence**.

Introduces the expected cost function used throughout this repo:
`cost = 10 × missed frauds + 1 × false alarms`. Splits before balancing. Its dataset is
only 1,000 claims at 24.7% fraud, which is far more balanced than a real claim stream.

**3. Olivia, D., Khan, Z. & Shetty, S. (2025).** *A Machine Learning and Explainable
Artificial Intelligence Approach for Insurance Fraud Classification.* **Inteligencia
Artificial**, 28(75), 140–169.
<https://journal.iberamia.org/index.php/intartif/article/view/1656>

Same dataset. Runs SHAP, LIME, ELI5 and QLattice over a stacked ensemble. Source of the
explainability check that caught the `PolicyNumber` problem. Leaves its meta-learner at
a fixed 0.5 threshold, which holds its recall to 0.43.

**4. Brant, S. B. & Hobæk Haff, I. (2023).** *The fraud loss for selecting the model
complexity in fraud detection.* **Journal of Applied Statistics**, 50(10), 2209–2227.
<https://arxiv.org/abs/2101.11907>

Defines the **fraud loss**: the number of non-fraudulent cases among the top *k*
reviewed. Starts from the same premise as this project — investigators can only review
a fixed number of cases. They use the fraud loss to select model complexity; this
project uses it to select the decision threshold, combined with paper 2's cost function.
Budgeted detection is established in tax and card fraud. It is absent from the auto
insurance literature, which is the gap this project fills.

---

## Dataset

Angoss Knowledge Seeker automobile insurance dataset (`carclaims.csv`), originally
released by EXL Service. Claims from January 1994 through December 1996.

| | |
|---|---|
| Records | 15,420 |
| Features | 33 |
| Legitimate | 14,497 (94%) |
| Fraudulent | 923 (6%) |

- <https://github.com/Rashmi-77/Vehicle-Insurance-Fraud-Detection> (cited by parent paper 1)
- <https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection>

## License

Dataset is publicly available. Parent paper 1 is CC BY.