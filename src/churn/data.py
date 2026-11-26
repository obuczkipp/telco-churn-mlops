"""Data ingestion, validation and cleaning.

Why a separate validation step?
In production, most model failures are not caused by the algorithm but by the
data: a renamed column, a new category, a numeric field that suddenly arrives
as text. Failing loudly at ingestion is far cheaper than silently producing bad
predictions, so `validate` checks the schema before anything else runs.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import resolve

log = logging.getLogger(__name__)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Allowed values per categorical column: the "data contract" of the model.
EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
}


class DataValidationError(ValueError):
    """Raised when incoming data violates the expected schema."""


def download(url: str, dest: str | Path, force: bool = False) -> Path:
    """Download the raw CSV once and cache it locally."""
    dest = resolve(str(dest))
    if dest.exists() and not force:
        log.info("Using cached file %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted, configured URL)
    return dest


def clean(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Fix the known quirks of the raw file.

    TotalCharges is stored as text and is blank for 11 customers. All of them
    have tenure == 0, i.e. they joined in the current month and have not been
    billed yet. The blank is therefore not "missing at random": the correct
    value is structurally 0, so we impute 0 rather than the mean or median
    (mean imputation would invent a billing history that does not exist).
    """
    df = df.copy()
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = total.isna()
    if blank.any():
        assert (df.loc[blank, "tenure"] == 0).all(), "Blank TotalCharges with tenure > 0"
        log.info("Imputing TotalCharges=0 for %d new customers (tenure=0)", int(blank.sum()))
    df["TotalCharges"] = total.fillna(0.0)
    if target in df.columns and df[target].dtype != "int64":
        df[target] = (df[target].astype(str).str.strip() == "Yes").astype("int64")
    return df


def validate(df: pd.DataFrame, require_target: bool = True, target: str = "Churn") -> None:
    """Check columns, types, ranges and category levels. Raise on any violation."""
    errors: list[str] = []
    required = ALL_FEATURES + ([target] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} is not numeric")
        elif df[col].isna().any():
            errors.append(f"{col} has {int(df[col].isna().sum())} missing values")
        elif (df[col] < 0).any():
            errors.append(f"{col} has negative values")

    for col, allowed in EXPECTED_CATEGORIES.items():
        unexpected = set(df[col].dropna().unique()) - allowed
        if unexpected:
            errors.append(f"{col} has unexpected levels {sorted(unexpected)}")

    if require_target and not set(df[target].unique()) <= {0, 1}:
        errors.append(f"{target} must be binary 0/1")

    if errors:
        raise DataValidationError("; ".join(errors))


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Download (if needed), clean and validate the full dataset."""
    path = download(cfg["data"]["url"], cfg["data"]["raw_path"])
    df = clean(pd.read_csv(path), target=cfg["data"]["target"])
    validate(df, target=cfg["data"]["target"])
    log.info("Loaded %d rows, churn rate %.3f", len(df), df[cfg["data"]["target"]].mean())
    return df


def split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split.

    Stratifying on the target keeps the churn rate (~26.5%) identical in both
    parts. With an imbalanced target, a purely random split can shift the base
    rate by a percentage point or more, which biases metrics such as
    precision and the Brier score that depend on prevalence.
    """
    target = cfg["data"]["target"]
    train, test = train_test_split(
        df,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=df[target],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
