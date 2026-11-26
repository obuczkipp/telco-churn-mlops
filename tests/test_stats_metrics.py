import numpy as np
import pandas as pd

from churn.features import TelcoFeatures
from churn.metrics import bootstrap_ci, expected_calibration_error, paired_bootstrap_diff
from churn.stats_analysis import churn_rate_ci, cramers_v, logistic_inference, univariate_tests


def test_cramers_v_bounds():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, 5000)
    independent = pd.crosstab(a, rng.integers(0, 2, 5000))
    perfect = pd.crosstab(a, a)
    assert cramers_v(independent) < 0.05
    assert cramers_v(perfect) > 0.99


def test_wilson_interval_contains_rate():
    rate, lo, hi = churn_rate_ci(pd.Series([1] * 30 + [0] * 70))
    assert lo < rate == 0.3 < hi


def test_univariate_tests_find_the_planted_signal(customers):
    res = univariate_tests(customers, "Churn").set_index("feature")
    assert res.loc["Contract", "significant"]
    assert res.loc["tenure", "effect_size"] < 0  # churners have shorter tenure
    assert res["p_adj_bh"].ge(res["p_value"]).all()  # BH never lowers p-values


def test_logistic_inference_outputs(customers):
    out = logistic_inference(customers, "Churn")
    assert out["odds_ratios"].loc["tenure_years", "odds_ratio"] < 1
    assert out["vif"]["VIF"].max() < 10


def test_features_handle_zero_tenure(customers):
    X = customers.copy()
    X.loc[0, ["tenure", "TotalCharges"]] = [0, 0.0]
    out = TelcoFeatures().transform(X)
    assert np.isfinite(out[["log_tenure", "avg_monthly_spend", "charge_gap"]]).all().all()
    assert out.loc[0, "charge_gap"] == 0


def test_ece_is_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(size=500) * 0.7, 0, 1)
    est, lo, hi = bootstrap_ci(y, p, "roc_auc", n_boot=200)
    assert lo <= est <= hi


def test_paired_bootstrap_identical_models():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.uniform(size=300)
    res = paired_bootstrap_diff(y, p, p, n_boot=100)
    assert res["diff"] == 0 and res["p_value"] == 1.0
