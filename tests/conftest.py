"""Synthetic data so unit tests run offline, fast and deterministically."""

import numpy as np
import pandas as pd
import pytest

from churn.data import EXPECTED_CATEGORIES


def make_customers(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(sorted(v), n) for c, v in EXPECTED_CATEGORIES.items()})
    df["customerID"] = [f"C{i:05d}" for i in range(n)]
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    # Churn depends on contract and tenure, so models have signal to learn.
    logit = -1 + 1.5 * (df["Contract"] == "Month-to-month") - 0.05 * df["tenure"]
    df["Churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int64")
    return df


@pytest.fixture
def customers() -> pd.DataFrame:
    return make_customers()
