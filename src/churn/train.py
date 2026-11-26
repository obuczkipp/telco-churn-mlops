"""Model training, selection, calibration and evaluation.

Workflow and the statistical reasoning behind each step
-------------------------------------------------------
1. Hold out a stratified 20% test set. It is used exactly once, at the end,
   so the reported performance is an unbiased estimate for new customers.

2. Compare candidate models with *nested* cross-validation on the training
   set. The inner loop (GridSearchCV) tunes hyperparameters; the outer loop
   (5-fold stratified CV) evaluates the whole "tune-then-fit" procedure.
   Tuning and evaluating on the same folds would reward the model that
   overfits the folds best, giving an optimistic score.

3. Candidates:
   * Prior baseline: always predicts the training churn rate. Every model
     must beat it; its AUC is 0.5 by construction.
   * Logistic regression (L2-regularised): linear in log-odds, interpretable,
     well calibrated by construction (it directly maximises the likelihood).
     C is the inverse regularisation strength: small C shrinks coefficients
     towards 0 (more bias, less variance).
   * Histogram gradient boosting: an ensemble of shallow trees fitted
     sequentially to the errors of the previous ones. Captures non-linear
     effects and interactions automatically, but its probabilities are often
     miscalibrated.
   We do NOT use class_weight="balanced" or oversampling (SMOTE). Re-weighting
   classes distorts the predicted probabilities (they no longer equal the
   real churn rate), and the retention decision needs true probabilities.
   Class imbalance is handled where it belongs: in the decision threshold.

4. Select the model by mean outer-fold ROC-AUC, then calibrate it with
   CalibratedClassifierCV (isotonic regression fitted on out-of-fold
   predictions). Isotonic calibration learns a monotone mapping from raw
   scores to observed frequencies, so it preserves the ranking (AUC barely
   changes) while fixing the probability scale.

5. Evaluate on the test set with bootstrap CIs, and test whether the gap
   between the two model families is statistically significant with a paired
   bootstrap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn.data import ALL_FEATURES
from churn.features import TelcoFeatures, build_preprocessor
from churn.metrics import bootstrap_ci, compute_metrics, paired_bootstrap_diff

log = logging.getLogger(__name__)

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
    "log_loss": "neg_log_loss",
}


def _pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("features", TelcoFeatures()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


def candidate_models(seed: int) -> dict[str, object]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": GridSearchCV(
            _pipeline(LogisticRegression(max_iter=5000)),
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            scoring="roc_auc", cv=inner, n_jobs=-1,
        ),
        "gradient_boosting": GridSearchCV(
            _pipeline(HistGradientBoostingClassifier(
                max_iter=300, l2_regularization=1.0, min_samples_leaf=40,
                early_stopping=False, random_state=seed)),
            param_grid={"model__learning_rate": [0.03, 0.1],
                        "model__max_leaf_nodes": [7, 15]},
            scoring="roc_auc", cv=inner, n_jobs=-1,
        ),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series, cfg: dict) -> pd.DataFrame:
    seed = cfg["split"]["random_state"]
    outer = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    rows = []
    for name, model in candidate_models(seed).items():
        log.info("Nested CV: %s", name)
        res = cross_validate(model, X, y, cv=outer, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for m in SCORING:
            vals = res[f"test_{m}"]
            if m in ("brier", "log_loss"):
                vals = -vals  # sklearn maximises, so these come back negated
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final(X: pd.DataFrame, y: pd.Series, model_name: str, cfg: dict):
    """Tune on the full training set, then wrap in a calibrator."""
    seed = cfg["split"]["random_state"]
    search = candidate_models(seed)[model_name]
    search.fit(X, y)
    best_params = search.best_params_
    log.info("Best params for %s: %s", model_name, best_params)
    uncalibrated = search.best_estimator_
    calibrated = CalibratedClassifierCV(
        clone(uncalibrated), method=cfg["model"]["calibration"],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    ).fit(X, y)
    return uncalibrated, calibrated, best_params


def evaluate(y: np.ndarray, p: np.ndarray, cfg: dict) -> dict[str, dict[str, float]]:
    n_boot = cfg["stats"]["n_bootstrap"]
    out = {}
    for m in ("roc_auc", "pr_auc", "brier", "ece"):
        est, lo, hi = bootstrap_ci(y, p, m, n_boot=n_boot)
        out[m] = {"estimate": est, "ci_low": lo, "ci_high": hi}
    out["log_loss"] = {"estimate": compute_metrics(y, p)["log_loss"]}
    return out


def run_training(train: pd.DataFrame, test: pd.DataFrame, cfg: dict, models_dir: Path) -> dict:
    target = cfg["data"]["target"]
    X_tr, y_tr = train[ALL_FEATURES], train[target]
    X_te, y_te = test[ALL_FEATURES], test[target].to_numpy()

    cv_table = cross_validate_models(X_tr, y_tr, cfg)
    contenders = cv_table[cv_table["model"] != "baseline_prior"]
    best_name = contenders.sort_values("roc_auc_mean", ascending=False).iloc[0]["model"]
    other_name = [m for m in contenders["model"] if m != best_name][0]
    log.info("Selected model: %s", best_name)

    best_raw, best_cal, best_params = fit_final(X_tr, y_tr, best_name, cfg)
    other_raw, other_cal, _ = fit_final(X_tr, y_tr, other_name, cfg)

    preds = {
        f"{best_name} (calibrated)": best_cal.predict_proba(X_te)[:, 1],
        f"{best_name} (raw)": best_raw.predict_proba(X_te)[:, 1],
        f"{other_name} (calibrated)": other_cal.predict_proba(X_te)[:, 1],
    }
    test_metrics = {name: evaluate(y_te, p, cfg) for name, p in preds.items()}
    comparison = paired_bootstrap_diff(
        y_te, preds[f"{best_name} (calibrated)"], preds[f"{other_name} (calibrated)"],
        metric="roc_auc", n_boot=cfg["stats"]["n_bootstrap"],
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_cal, models_dir / "model.joblib")
    train[ALL_FEATURES + [target]].to_csv(models_dir / "reference.csv", index=False)
    meta = {
        "model_name": best_name,
        "best_params": best_params,
        "calibration": cfg["model"]["calibration"],
        "features": ALL_FEATURES,
        "n_train": len(train),
        "train_churn_rate": float(y_tr.mean()),
        "test_roc_auc": test_metrics[f"{best_name} (calibrated)"]["roc_auc"]["estimate"],
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return {
        "cv_table": cv_table,
        "best_name": best_name,
        "other_name": other_name,
        "best_params": best_params,
        "predictions": preds,
        "test_metrics": test_metrics,
        "auc_comparison": comparison,
        "model": best_cal,
        "metadata": meta,
    }
