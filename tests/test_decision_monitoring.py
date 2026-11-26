import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import decision, monitoring
from churn.config import load_config
from churn.data import ALL_FEATURES
from churn.features import TelcoFeatures, build_preprocessor

BIZ = {"horizon_months": 12, "offer_cost": 60.0, "contact_cost": 5.0, "save_rate": 0.3,
       "save_rate_grid": [0.1, 0.3, 0.5]}


def test_expected_value_break_even():
    # Contact iff p > cost / (s * CLV). Here CLV = 100*12 = 1200, cost = 65, s = 0.3
    threshold = 65 / (0.3 * 1200)
    clv = decision.customer_value(np.array([100.0, 100.0]), 12)
    ev = decision.expected_value(np.array([threshold - 0.01, threshold + 0.01]), clv, 0.3, 65)
    assert ev[0] < 0 < ev[1]


def test_policy_comparison_sanity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = np.clip(0.6 * y + 0.4 * rng.uniform(size=1000), 0, 1)
    mc = rng.uniform(20, 120, 1000)
    res = decision.policy_comparison(y, p, mc, BIZ).set_index("policy")
    assert res.loc["Contact nobody", "net_value"] == 0
    assert res.loc["Contact everybody", "recall"] == 1
    # With an informative score, the EV rule beats contacting everybody.
    assert res.loc["Expected-value rule (EV > 0)", "net_value"] >= res.loc["Contact everybody", "net_value"]


def test_gains_table_lift_decreases():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + 0.5 * rng.uniform(size=2000), 0, 1)
    g = decision.gains_table(y, p)
    assert len(g) == 10
    assert g["lift"].iloc[0] > g["lift"].iloc[-1]
    assert np.isclose(g["cumulative_recall"].iloc[-1], 1.0)


def test_psi_detects_shift_but_not_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert monitoring.psi_numeric(ref, rng.normal(0, 1, 5000)) < 0.02
    assert monitoring.psi_numeric(ref, rng.normal(1, 1, 5000)) > 0.25
    a = pd.Series(["x"] * 500 + ["y"] * 500)
    assert monitoring.psi_categorical(a, a) == 0


def test_monitoring_flags_simulated_drift(customers):
    cfg = load_config()
    ref = customers
    shifted = monitoring.simulate_drift(customers, weight_new=6)
    assert shifted["tenure"].mean() < ref["tenure"].mean()
    drift = monitoring.feature_drift(ref, shifted, cfg).set_index("feature")
    assert drift.loc["tenure", "status"] != "OK"
    assert monitoring.feature_drift(ref, ref, cfg)["status"].eq("OK").all()


def test_end_to_end_pipeline_predicts_probabilities(customers):
    model = Pipeline([("f", TelcoFeatures()), ("p", build_preprocessor()),
                      ("m", LogisticRegression(max_iter=2000))])
    model.fit(customers[ALL_FEATURES], customers["Churn"])
    new = customers[ALL_FEATURES].head(5).copy()
    new.loc[new.index[0], "PaymentMethod"] = "Crypto"  # unseen level must not crash scoring
    p = model.predict_proba(new)[:, 1]
    assert ((p > 0) & (p < 1)).all()
