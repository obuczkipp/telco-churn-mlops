"""Production monitoring: data drift, prediction drift and performance decay.

A model is trained on yesterday's customers and applied to today's. When the
customer base, prices or products change, the model's inputs drift away from
what it learned and its accuracy silently degrades. Monitoring compares each
new scoring batch with a *reference* sample (the training data).

Population Stability Index (PSI)
--------------------------------
PSI = sum_b (a_b - e_b) * ln(a_b / e_b)
where e_b and a_b are the shares of the reference ("expected") and current
("actual") data in bin b. Bins come from reference quantiles for numeric
features, or are the category levels for categorical ones. PSI is a
symmetrised KL divergence. Industry rule of thumb (credit risk origin):
  < 0.10  no significant shift
  0.10 - 0.25  moderate shift, investigate
  > 0.25  major shift, model likely needs retraining

Statistical tests alongside PSI
-------------------------------
* Kolmogorov-Smirnov (numeric): H0 = both samples come from the same
  distribution. Statistic = largest gap between the two empirical CDFs.
* Chi-square (categorical): H0 = category shares are the same.
With large batches these tests flag even negligible shifts as significant,
which is why decisions are based on PSI (an effect size) and the p-values are
reported as supporting evidence only.

Performance monitoring
----------------------
Churn labels arrive with a delay (we only know a customer churned at the end
of the observation window). When labels are available, live AUC and
calibration are compared with the values measured at training time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from churn.data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPS = 1e-4  # avoids log(0) for empty bins


def psi_numeric(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(cur, edges)[0] / len(cur)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    levels = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    e = ref.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    a = cur.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(psi: float, warn: float, alert: float) -> str:
    return "ALERT" if psi > alert else "WARN" if psi > warn else "OK"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg["monitoring"]
    rows = []
    for col in NUMERIC_FEATURES:
        psi = psi_numeric(ref[col], cur[col], m["psi_bins"])
        ks = stats.ks_2samp(ref[col], cur[col])
        rows.append({"feature": col, "type": "numeric", "psi": psi, "test": "KS",
                     "statistic": ks.statistic, "p_value": ks.pvalue,
                     "ref_mean": ref[col].mean(), "cur_mean": cur[col].mean()})
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        psi = psi_categorical(ref[col], cur[col])
        levels = sorted(set(ref[col].unique()) | set(cur[col].unique()), key=str)
        table = np.array([
            ref[col].value_counts().reindex(levels, fill_value=0).to_numpy(),
            cur[col].value_counts().reindex(levels, fill_value=0).to_numpy(),
        ])
        table = table[:, table.sum(axis=0) > 0]
        chi2, p, _, _ = stats.chi2_contingency(table)
        rows.append({"feature": col, "type": "categorical", "psi": psi, "test": "chi-square",
                     "statistic": chi2, "p_value": p, "ref_mean": np.nan, "cur_mean": np.nan})
    out = pd.DataFrame(rows)
    out["status"] = [_status(v, m["psi_warn"], m["psi_alert"]) for v in out["psi"]]
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_drift(df: pd.DataFrame, weight_new: float = 4.0, seed: int = 7) -> pd.DataFrame:
    """Create a "next month" batch with a known covariate shift, for demonstration.

    Scenario: an acquisition campaign brought in many new customers, so the
    scoring batch over-represents customers in their first year.

    Implementation: weighted resampling of *real* rows (customers with
    tenure <= 12 are drawn `weight_new` times more often). Editing feature
    values directly (e.g. "add 12% to prices") would break the link between
    features and the observed churn label, making any performance check on
    that batch meaningless. Resampling keeps every row internally consistent:
    P(churn | features) is unchanged and only P(features) shifts. This is
    exactly the definition of covariate shift, the most common kind of drift.

    Expected result: tenure, TotalCharges and Contract are flagged, predicted
    and observed churn both rise, and AUC stays roughly stable (the model is
    still right *given* the features, the population is just different).
    """
    rng = np.random.default_rng(seed)
    w = np.where(df["tenure"] <= 12, weight_new, 1.0)
    idx = rng.choice(len(df), size=len(df), replace=True, p=w / w.sum())
    return df.iloc[idx].reset_index(drop=True)


def overall_status(drift: pd.DataFrame, perf: dict | None, cfg: dict) -> str:
    if perf and perf.get("auc_drop", 0) > cfg["monitoring"]["auc_drop_alert"]:
        return "ALERT"
    if (drift["status"] == "ALERT").any():
        return "ALERT"
    if (drift["status"] == "WARN").any():
        return "WARN"
    return "OK"
