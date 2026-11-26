"""Feature engineering and preprocessing pipelines.

Everything is wrapped in a scikit-learn Pipeline so that the exact same
transformation is applied in training, cross-validation and production
scoring. Fitting the scaler or encoder on the full dataset before splitting
would leak information from the test set into training ("data leakage") and
make the evaluation over-optimistic; inside a Pipeline the transformers are
re-fitted on each training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]

ENGINEERED_NUMERIC = ["log_tenure", "avg_monthly_spend", "n_addon_services", "charge_gap"]


class TelcoFeatures(BaseEstimator, TransformerMixin):
    """Domain-driven features, computed row by row (no fitted state, so no leakage).

    * log_tenure: churn risk falls steeply in the first months and then
      flattens, so the relationship with tenure is roughly logarithmic. A
      linear model cannot learn this curve from raw tenure alone.
    * avg_monthly_spend: TotalCharges / tenure, the historical average bill.
    * charge_gap: current bill minus historical average. A positive gap means
      the customer is paying more than they used to (price increase or
      upsell), a plausible churn trigger.
    * n_addon_services: number of add-on services. More services = higher
      switching cost ("stickiness").
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["log_tenure"] = np.log1p(X["tenure"])
        safe_tenure = X["tenure"].where(X["tenure"] > 0, 1)
        X["avg_monthly_spend"] = np.where(
            X["tenure"] > 0, X["TotalCharges"] / safe_tenure, X["MonthlyCharges"]
        )
        X["charge_gap"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        X["n_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + ENGINEERED_NUMERIC, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features, one-hot encode categoricals.

    * StandardScaler: puts numeric features on a common scale, so the L2
      penalty in logistic regression treats all coefficients equally. Tree
      models do not need it, but it does no harm.
    * OneHotEncoder(handle_unknown="ignore"): a category never seen in
      training (e.g. a new payment method) becomes an all-zero vector instead
      of crashing the scoring job. Monitoring flags such cases separately.
    * drop="if_binary": Yes/No columns become a single 0/1 column, avoiding
      redundant perfectly-correlated pairs.
    """
    numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
