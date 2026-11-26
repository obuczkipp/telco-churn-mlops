# Model evaluation and retention decision

## 1. Model selection: nested 5-fold cross-validation (training set)

Mean ± standard deviation across outer folds. Hyperparameters tuned in an inner 3-fold loop.

| model | ROC-AUC | PR-AUC | Brier | Log-loss |
|---|---|---|---|---|
| baseline_prior | 0.500 ± 0.000 | 0.265 ± 0.000 | 0.1949 ± 0.0000 | 0.5786 ± 0.0001 |
| logistic_regression | 0.850 ± 0.013 | 0.671 ± 0.016 | 0.1333 ± 0.0052 | 0.4112 ± 0.0124 |
| gradient_boosting | 0.848 ± 0.013 | 0.662 ± 0.027 | 0.1346 ± 0.0062 | 0.4141 ± 0.0142 |

Selected: **logistic_regression**, parameters `{'model__C': 10.0}`, then isotonic calibration.

## 2. Held-out test set (bootstrap 95% CI)

| model | roc_auc | pr_auc | brier | ece |
|---|---|---|---|---|
| logistic_regression (calibrated) | 0.845 [0.824, 0.865] | 0.656 [0.614, 0.697] | 0.136 [0.127, 0.145] | 0.019 [0.016, 0.044] |
| logistic_regression (raw) | 0.846 [0.825, 0.866] | 0.661 [0.620, 0.703] | 0.135 [0.127, 0.144] | 0.020 [0.018, 0.045] |
| gradient_boosting (calibrated) | 0.846 [0.824, 0.866] | 0.661 [0.619, 0.706] | 0.135 [0.126, 0.144] | 0.021 [0.018, 0.044] |

Paired bootstrap, ROC-AUC difference logistic_regression minus gradient_boosting: **-0.0009** (95% CI -0.0070 to +0.0056, p = 0.808).

## 3. Gains / lift by decile (selected model, test set)

| decile | customers | churners | mean_score | churn_rate | lift | cumulative_recall |
|---|---|---|---|---|---|---|
| 1 | 141 | 108 | 0.79 | 0.77 | 2.89 | 0.29 |
| 2 | 141 | 79 | 0.59 | 0.56 | 2.11 | 0.50 |
| 3 | 141 | 58 | 0.44 | 0.41 | 1.55 | 0.66 |
| 4 | 141 | 43 | 0.31 | 0.30 | 1.15 | 0.77 |
| 5 | 141 | 38 | 0.24 | 0.27 | 1.02 | 0.87 |
| 6 | 141 | 26 | 0.15 | 0.18 | 0.69 | 0.94 |
| 7 | 141 | 15 | 0.09 | 0.11 | 0.40 | 0.98 |
| 8 | 141 | 4 | 0.04 | 0.03 | 0.11 | 0.99 |
| 9 | 141 | 1 | 0.02 | 0.01 | 0.03 | 0.99 |
| 10 | 140 | 2 | 0.00 | 0.01 | 0.05 | 1.00 |

## 4. Targeting policies (test set)

Assumptions: customer value = 12 months of charges, offer cost $60 + contact cost $5, save rate 30%.

| policy | contacted | share_contacted | churners_reached | recall | precision | net_value | net_value_per_1000_customers |
|---|---|---|---|---|---|---|---|
| Contact nobody | 0 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| Contact everybody | 1409 | 1.00 | 374 | 1.00 | 0.27 | 6388.64 | 4534.17 |
| Fixed threshold p > 0.5 | 304 | 0.22 | 198 | 0.53 | 0.65 | 35464.54 | 25170.01 |
| Top 20% by churn probability | 283 | 0.20 | 188 | 0.50 | 0.66 | 34011.64 | 24138.85 |
| Expected-value rule (EV > 0) | 545 | 0.39 | 272 | 0.73 | 0.50 | 44340.74 | 31469.65 |

## 5. Sensitivity to the save-rate assumption

| save_rate | share_contacted | net_value_ev_rule | net_value_contact_all |
|---|---|---|---|
| 0.10 | 0.10 | 1609.62 | -58927.12 |
| 0.20 | 0.30 | 18944.04 | -26269.24 |
| 0.30 | 0.39 | 44340.74 | 6388.64 |
| 0.40 | 0.45 | 71207.60 | 39046.52 |
| 0.50 | 0.49 | 99982.10 | 71704.40 |
