"""Evaluation metrics and uncertainty estimates.

Why several metrics?
* ROC-AUC: probability that a random churner gets a higher score than a random
  non-churner. Measures *ranking* quality, independent of the threshold and of
  the base rate. 0.5 = random, 1.0 = perfect.
* PR-AUC (average precision): area under the precision-recall curve. More
  informative than ROC-AUC when the positive class is the minority, because it
  focuses on how clean the top of the ranking is. Its random baseline equals
  the churn rate (~0.27), not 0.5.
* Brier score: mean squared error of the predicted probabilities,
  mean((p - y)^2). Rewards probabilities that are both sharp and calibrated.
  Lower is better; a model that always predicts the base rate scores
  p(1-p) ~ 0.195.
* Log-loss: negative log-likelihood. Punishes confident wrong predictions
  heavily.
* Expected Calibration Error (ECE): average gap between predicted
  probability and observed churn rate across probability bins. The retention
  decision multiplies the probability by money, so a model predicting 40%
  should be right about 40% of the time.

Why bootstrap confidence intervals?
The test set is one sample of ~1400 customers. A different sample would give a
different AUC. The bootstrap resamples the test set with replacement many
times, recomputes the metric, and uses the 2.5th and 97.5th percentiles as a
95% CI (percentile bootstrap). This tells stakeholders how precise a reported
number is, and whether two models genuinely differ.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b (n_b / n) * |mean(y_b) - mean(p_b)|."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier": brier_score_loss,
    "log_loss": lambda y, p: log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
    "ece": expected_calibration_error,
}


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {name: float(fn(y, p)) for name, fn in METRICS.items()}


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, metric: str, n_boot: int = 1000,
    alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and percentile-bootstrap CI for one metric.

    Resamples are stratified by class so every resample contains both classes
    (otherwise ROC-AUC is undefined) and keeps the original churn rate.
    """
    y, p = np.asarray(y), np.asarray(p)
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats_ = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats_[i] = fn(y[idx], p[idx])
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(y, p)), float(lo), float(hi)


def paired_bootstrap_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str = "roc_auc",
    n_boot: int = 1000, seed: int = 42,
) -> dict[str, float]:
    """Is model A really better than model B on this metric?

    Both models are evaluated on the *same* resampled customers in each
    iteration (paired design). Pairing removes the shared sample noise, so the
    test is much more powerful than comparing two separate CIs for overlap.
    The two-sided p-value is approximated as twice the share of resamples in
    which the difference has the opposite sign to the observed one.
    """
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[i] = fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx])
    observed = fn(y, p_a) - fn(y, p_b)
    if observed >= 0:
        p_val = 2 * np.mean(diffs <= 0)
    else:
        p_val = 2 * np.mean(diffs >= 0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p_val))}
