# Monitoring report

Reference = training data. Two scoring batches are checked:
* **Batch A**: the untouched test set (expected result: no drift, no false alarm).
* **Batch B**: a simulated covariate shift. The test set is resampled so that
  customers in their first year are 4x over-represented (an acquisition campaign).
  Rows are real, so labels stay consistent and the performance check is valid.

## Batch A no_drift: overall status **OK**

Prediction PSI = 0.007; mean predicted churn 26.7% vs. reference 26.4%. Live AUC 0.845 (baseline 0.845); ECE 0.019.

No feature exceeded the PSI warning level.

<details><summary>All features</summary>

| feature | type | psi | status | test | statistic | p_value |
|---|---|---|---|---|---|---|
| MonthlyCharges | numeric | 0.012 | OK | KS | 0.032 | 0.191 |
| TotalCharges | numeric | 0.008 | OK | KS | 0.030 | 0.271 |
| tenure | numeric | 0.007 | OK | KS | 0.021 | 0.683 |
| OnlineBackup | categorical | 0.004 | OK | chi-square | 4.937 | 0.085 |
| MultipleLines | categorical | 0.003 | OK | chi-square | 3.355 | 0.187 |
| StreamingTV | categorical | 0.003 | OK | chi-square | 3.127 | 0.209 |
| PhoneService | categorical | 0.002 | OK | chi-square | 1.698 | 0.193 |
| StreamingMovies | categorical | 0.001 | OK | chi-square | 1.153 | 0.562 |
| TechSupport | categorical | 7.77e-04 | OK | chi-square | 0.870 | 0.647 |
| DeviceProtection | categorical | 4.45e-04 | OK | chi-square | 0.501 | 0.779 |
| PaymentMethod | categorical | 4.04e-04 | OK | chi-square | 0.453 | 0.929 |
| gender | categorical | 3.67e-04 | OK | chi-square | 0.376 | 0.540 |
| OnlineSecurity | categorical | 3.38e-04 | OK | chi-square | 0.381 | 0.827 |
| Dependents | categorical | 2.95e-04 | OK | chi-square | 0.297 | 0.586 |
| SeniorCitizen | categorical | 2.44e-04 | OK | chi-square | 0.232 | 0.630 |
| InternetService | categorical | 2.36e-04 | OK | chi-square | 0.267 | 0.875 |
| Partner | categorical | 1.82e-04 | OK | chi-square | 0.179 | 0.672 |
| Contract | categorical | 1.44e-04 | OK | chi-square | 0.163 | 0.922 |
| PaperlessBilling | categorical | 1.01e-04 | OK | chi-square | 0.094 | 0.759 |

</details>

## Batch B simulated_drift: overall status **ALERT**

Prediction PSI = 0.158; mean predicted churn 36.8% vs. reference 26.4%. Live AUC 0.809 (baseline 0.845); ECE 0.044.

Flagged features:

| feature | psi | status | test | p_value | ref_mean | cur_mean |
|---|---|---|---|---|---|---|
| tenure | 0.510 | ALERT | KS | 1.16e-124 | 32.485 | 17.847 |
| TotalCharges | 0.406 | ALERT | KS | 5.87e-96 | 2299.335 | 1192.910 |
| Contract | 0.121 | WARN | chi-square | 1.13e-27 |  |  |

<details><summary>All features</summary>

| feature | type | psi | status | test | statistic | p_value |
|---|---|---|---|---|---|---|
| tenure | numeric | 0.510 | ALERT | KS | 0.352 | 1.16e-124 |
| TotalCharges | numeric | 0.406 | ALERT | KS | 0.309 | 5.87e-96 |
| Contract | categorical | 0.121 | WARN | chi-square | 124.087 | 1.13e-27 |
| MonthlyCharges | numeric | 0.097 | OK | KS | 0.124 | 1.66e-15 |
| OnlineBackup | categorical | 0.091 | OK | chi-square | 93.676 | 4.56e-21 |
| PaymentMethod | categorical | 0.077 | OK | chi-square | 83.292 | 6.04e-18 |
| MultipleLines | categorical | 0.069 | OK | chi-square | 77.138 | 1.78e-17 |
| Partner | categorical | 0.066 | OK | chi-square | 72.331 | 1.82e-17 |
| OnlineSecurity | categorical | 0.063 | OK | chi-square | 64.347 | 1.06e-14 |
| StreamingTV | categorical | 0.062 | OK | chi-square | 66.412 | 3.79e-15 |
| DeviceProtection | categorical | 0.060 | OK | chi-square | 63.556 | 1.58e-14 |
| TechSupport | categorical | 0.047 | OK | chi-square | 49.291 | 1.98e-11 |
| StreamingMovies | categorical | 0.041 | OK | chi-square | 44.369 | 2.32e-10 |
| Dependents | categorical | 0.010 | OK | chi-square | 10.713 | 0.001 |
| InternetService | categorical | 0.005 | OK | chi-square | 6.152 | 0.046 |
| PhoneService | categorical | 0.004 | OK | chi-square | 4.163 | 0.041 |
| SeniorCitizen | categorical | 0.004 | OK | chi-square | 3.941 | 0.047 |
| gender | categorical | 1.46e-04 | OK | chi-square | 0.141 | 0.707 |
| PaperlessBilling | categorical | 3.41e-05 | OK | chi-square | 0.027 | 0.868 |

</details>
