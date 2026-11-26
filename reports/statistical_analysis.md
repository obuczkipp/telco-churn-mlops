# Statistical analysis of churn drivers

Customers: **5634**. Churn rate: **26.5%** (95% Wilson CI 25.4% to 27.7%).

## 1. Univariate tests (Benjamini-Hochberg corrected)

Chi-square tests for categorical features (effect size: bias-corrected Cramér's V),
Mann-Whitney U for numeric features (effect size: rank-biserial r; negative = churners
have lower values). Sorted by absolute effect size. Rough guide for Cramér's V:
0.1 small, 0.3 medium, 0.5 large.

| feature | test | statistic | p_adj_bh | effect_size | effect_measure | min_expected_count | highest_churn_group | lowest_churn_group | significant |
|---|---|---|---|---|---|---|---|---|---|
| tenure | Mann-Whitney U | 1.64e+06 | 1.02e-159 | -0.470 | rank-biserial r |  | median churn=10.0 | median stay=38.0 | True |
| Contract | chi-square | 952.308 | 3.07e-206 | 0.411 | Cramer's V | 311.259 | Month-to-month (42.7%) | Two year (2.9%) | True |
| OnlineSecurity | chi-square | 694.281 | 1.10e-150 | 0.351 | Cramer's V | 322.139 | No (41.9%) | No internet service (7.2%) | True |
| TechSupport | chi-square | 670.306 | 1.32e-145 | 0.344 | Cramer's V | 322.139 | No (41.8%) | No internet service (7.2%) | True |
| InternetService | chi-square | 600.847 | 1.28e-130 | 0.326 | Cramer's V | 322.139 | Fiber optic (42.1%) | No (7.2%) | True |
| PaymentMethod | chi-square | 545.214 | 2.40e-117 | 0.310 | Cramer's V | 321.873 | Electronic check (45.7%) | Credit card (automatic) (14.9%) | True |
| OnlineBackup | chi-square | 488.038 | 2.87e-106 | 0.294 | Cramer's V | 322.139 | No (40.1%) | No internet service (7.2%) | True |
| TotalCharges | Mann-Whitney U | 2.19e+06 | 2.44e-62 | -0.291 | rank-biserial r |  | median churn=740.3 | median stay=1686.2 | True |
| DeviceProtection | chi-square | 441.284 | 3.56e-96 | 0.279 | Cramer's V | 322.139 | No (39.0%) | No internet service (7.2%) | True |
| MonthlyCharges | Mann-Whitney U | 3.86e+06 | 9.24e-46 | 0.248 | rank-biserial r |  | median churn=80.0 | median stay=64.4 | True |
| StreamingMovies | chi-square | 302.430 | 4.49e-66 | 0.231 | Cramer's V | 322.139 | No (33.6%) | No internet service (7.2%) | True |
| StreamingTV | chi-square | 299.017 | 2.23e-65 | 0.230 | Cramer's V | 322.139 | No (33.1%) | No internet service (7.2%) | True |
| PaperlessBilling | chi-square | 220.834 | 9.42e-50 | 0.198 | Cramer's V | 611.108 | Yes (33.8%) | No (16.0%) | True |
| Dependents | chi-square | 157.992 | 4.22e-36 | 0.167 | Cramer's V | 445.528 | No (31.4%) | Yes (15.2%) | True |
| Partner | chi-square | 119.629 | 9.66e-28 | 0.145 | Cramer's V | 724.149 | No (32.8%) | Yes (19.9%) | True |
| SeniorCitizen | chi-square | 119.436 | 9.98e-28 | 0.145 | Cramer's V | 244.125 | 1 (41.1%) | 0 (23.7%) | True |
| MultipleLines | chi-square | 10.975 | 0.005 | 0.040 | Cramer's V | 148.332 | Yes (28.8%) | No phone service (24.2%) | True |
| PhoneService | chi-square | 1.811 | 0.188 | 0.012 | Cramer's V | 148.332 | Yes (26.8%) | No (24.2%) | False |
| gender | chi-square | 0.027 | 0.868 | 0 | Cramer's V | 743.254 | Female (26.6%) | Male (26.4%) | False |

## 2. Multivariable logistic regression

n = 5634, McFadden pseudo R² = **0.283**, likelihood-ratio test vs. null model p < 1e-300, AIC = 4714.

Odds ratio > 1 increases churn odds, < 1 decreases them, holding all other
factors constant. Reference groups: month-to-month contract, DSL internet,
electronic check. Tenure is per year. MonthlyCharges and TotalCharges are excluded
(collinear with the service dummies and tenure, see VIF table).

| feature | odds_ratio | or_ci_low | or_ci_high | p_value | significant |
|---|---|---|---|---|---|
| internet_fiber | 2.161 | 1.806 | 2.586 | 3.90e-17 | True |
| paperlessbilling | 1.455 | 1.236 | 1.712 | 6.27e-06 | True |
| has_StreamingMovies | 1.343 | 1.128 | 1.599 | 9.41e-04 | True |
| has_StreamingTV | 1.336 | 1.122 | 1.591 | 0.001 | True |
| multiple_lines | 1.305 | 1.103 | 1.545 | 0.002 | True |
| senior_citizen | 1.175 | 0.976 | 1.415 | 0.088 | False |
| partner | 1.023 | 0.863 | 1.212 | 0.792 | False |
| has_DeviceProtection | 1.003 | 0.844 | 1.193 | 0.970 | False |
| has_OnlineBackup | 0.882 | 0.746 | 1.042 | 0.140 | False |
| dependents | 0.790 | 0.649 | 0.962 | 0.019 | True |
| pay_mailed_check | 0.750 | 0.608 | 0.925 | 0.007 | True |
| has_TechSupport | 0.702 | 0.582 | 0.846 | 2.11e-04 | True |
| tenure_years | 0.682 | 0.641 | 0.725 | 7.15e-34 | True |
| pay_bank_auto | 0.671 | 0.545 | 0.826 | 1.65e-04 | True |
| has_OnlineSecurity | 0.665 | 0.552 | 0.799 | 1.48e-05 | True |
| pay_card_auto | 0.654 | 0.527 | 0.810 | 1.04e-04 | True |
| contract_one_year | 0.489 | 0.386 | 0.619 | 2.76e-09 | True |
| internet_none | 0.377 | 0.282 | 0.505 | 5.61e-11 | True |
| contract_two_year | 0.253 | 0.173 | 0.371 | 1.42e-12 | True |

### Multicollinearity check (VIF)

VIF > 10 would indicate unstable coefficients; VIF > 5 deserves caution.

| feature | VIF |
|---|---|
| tenure_years | 2.77 |
| contract_two_year | 2.62 |
| internet_none | 2.49 |
| internet_fiber | 1.74 |
| pay_mailed_check | 1.70 |
| has_StreamingMovies | 1.65 |
| has_StreamingTV | 1.64 |
| contract_one_year | 1.63 |
| pay_card_auto | 1.53 |
| pay_bank_auto | 1.52 |
| has_DeviceProtection | 1.49 |
| has_TechSupport | 1.49 |
| partner | 1.45 |
| has_OnlineSecurity | 1.41 |
| has_OnlineBackup | 1.39 |
| dependents | 1.37 |
| multiple_lines | 1.34 |
| paperlessbilling | 1.21 |
| senior_citizen | 1.15 |
