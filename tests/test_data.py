import pandas as pd
import pytest

from churn.data import DataValidationError, clean, validate


def test_clean_imputes_blank_total_charges_as_zero_for_new_customers(customers):
    raw = customers.copy()
    raw["TotalCharges"] = raw["TotalCharges"].astype(str)
    raw["Churn"] = raw["Churn"].map({1: "Yes", 0: "No"})
    raw.loc[0, ["tenure", "TotalCharges"]] = [0, " "]
    out = clean(raw)
    assert out.loc[0, "TotalCharges"] == 0.0
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert set(out["Churn"].unique()) <= {0, 1}


def test_validate_accepts_clean_data(customers):
    validate(customers)


def test_validate_rejects_unknown_category(customers):
    bad = customers.copy()
    bad.loc[0, "PaymentMethod"] = "Crypto"
    with pytest.raises(DataValidationError, match="PaymentMethod"):
        validate(bad)


def test_validate_rejects_missing_column(customers):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(customers.drop(columns="Contract"))


def test_validate_rejects_negative_values(customers):
    bad = customers.copy()
    bad.loc[0, "MonthlyCharges"] = -5
    with pytest.raises(DataValidationError, match="negative"):
        validate(bad)
