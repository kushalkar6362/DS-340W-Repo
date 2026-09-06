# Automobile Insurance Claim Fraud Detection

Course project reproducing and extending a peer-reviewed fraud detection paper.

**Author:** Kushal
**Course:** Computational Data Science, Penn State

---

## Parent paper

Özaltın, Ö. & Karadağ Erdemir, Ö. (2025). *Detecting automobile insurance fraud using a novel penalty-driven feature selection method with particle swarm optimization and machine learning classifiers.* **Scientific Reports**, 15, 41793.

- Paper: https://www.nature.com/articles/s41598-025-25700-2
- PDF: https://www.nature.com/articles/s41598-025-25700-2.pdf
- DOI: 10.1038/s41598-025-25700-2
- Open access, CC BY

---

## Dataset

Angoss Knowledge Seeker automobile insurance dataset (`carclaims.csv`). Claims from January 1994 through December 1996.

| | |
|---|---|
| Records | 15,420 |
| Features | 33 |
| Legitimate | 14,497 (94%) |
| Fraudulent | 923 (6%) |

Sources:
- https://github.com/Rashmi-77/Vehicle-Insurance-Fraud-Detection (cited by the paper)
- https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection

---

## Repository layout

```
.
├── data/
│   └── carclaims.csv          # raw dataset
├── splits/
│   ├── paper/                 # train/test/validation, paper mode
│   └── clean/                 # train/test/validation, leakage-free mode
├── notebooks/
│   └── reproduction.ipynb     # baseline notebook being executed
├── split_data.py              # generates the three splits
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Generating the splits

70 / 20 / 10, stratified on the fraud label, seed fixed at 42.

```bash
python split_data.py --input data/carclaims.csv --mode paper
python split_data.py --input data/carclaims.csv --mode clean
```

| Split | Share | Purpose |
|---|---|---|
| Training | 70% | Model fitting, feature selection, grid search, five-fold CV |
| Test | 20% | Hyperparameter iteration, comparison against the paper's tables |
| Validation | 10% | Held out until the pipeline is bug free. Scored exactly once. |

The validation set is not read by any training or tuning code.

---

## Two split modes

The parent paper applies SMOTE to the full dataset *before* splitting. Synthetic minority points interpolated from training rows can therefore appear in the test set, which inflates the reported 97.55% accuracy.

- **`--mode paper`** balances first, then splits. Reproduces the published pipeline so the numbers are comparable.
- **`--mode clean`** splits first, then applies SMOTE to the training set only. Test and validation keep the true ~6% fraud rate.

Phase 1 runs paper mode. Phase 2 runs clean mode. Both results are reported.

---

## Method being reproduced

1. Encode categorical variables; standardize
2. Balance classes with SMOTE
3. Split into train / test / validation
4. Feature selection with PDFS-PSO — particle swarm optimization with a fitness function that penalizes selecting feature pairs whose absolute Pearson correlation exceeds a threshold α
5. Four thresholds tested: α = 0.85, 0.75, 0.65, 0.50
6. Eleven classifiers tuned by grid search with five-fold CV: Random Forest, SVM, KNN, Logistic Regression, Decision Tree, ANN, Gradient Boosting, AdaBoost, CatBoost, LightGBM, Stacking
7. Stacking classifier: SVM + Random Forest + Logistic Regression as level-0, Logistic Regression as meta-learner
8. Report accuracy, precision, recall, F1, MCC, AUC, log loss

Paper's best reported result: α = 0.65, 16 features retained, stacking classifier.

---

## Planned extension

Wrap the trained scorer in an agent layer: an LLM tool-calling service that accepts a claim record, calls the fraud model, retrieves the relevant policy language, and returns a scored triage recommendation with citations for the adjuster.

Peer-reviewed precedent: Arconzo et al. (2026), *Integrating LLMs With Computer Vision: A Geometrically Invariant System for First Notice of Loss in Car Insurance*, Expert Systems 43. https://onlinelibrary.wiley.com/doi/10.1111/exsy.70340

---

## License

Dataset is publicly available. Parent paper is CC BY.
