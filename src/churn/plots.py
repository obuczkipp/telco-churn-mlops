"""Figures for the README and for stakeholder presentations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

BLUE, ORANGE, GREY, GREEN = "#2b6cb0", "#dd6b20", "#a0aec0", "#2f855a"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def churn_by_group(df: pd.DataFrame, target: str, out: Path) -> None:
    """Churn rate with 95% Wilson CIs for the strongest categorical drivers."""
    from churn.stats_analysis import churn_rate_ci

    cols = ["Contract", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, col in zip(axes[:3], cols, strict=True):
        g = []
        for level, sub in df.groupby(col):
            r, lo, hi = churn_rate_ci(sub[target])
            g.append((str(level), r, lo, hi))
        g.sort(key=lambda t: t[1])
        labels = [t[0].replace(" (automatic)", "\n(auto)") for t in g]
        rates = np.array([t[1] for t in g])
        err = np.array([[t[1] - t[2] for t in g], [t[3] - t[1] for t in g]])
        ax.barh(labels, rates, xerr=err, color=BLUE, capsize=3)
        ax.axvline(df[target].mean(), color=ORANGE, ls="--", lw=1, label="overall")
        ax.set_title(col)
        ax.set_xlabel("churn rate")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    # Tenure buckets
    ax = axes[3]
    buckets = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72],
                     labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    rate = df.groupby(buckets, observed=True)[target].mean()
    ax.bar(rate.index.astype(str), rate.values, color=BLUE)
    ax.axhline(df[target].mean(), color=ORANGE, ls="--", lw=1)
    ax.set_title("Tenure (months)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].legend(loc="lower right", fontsize=8)
    _save(fig, out)


def odds_ratio_forest(or_table: pd.DataFrame, out: Path) -> None:
    t = or_table.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1))
    colors = [ORANGE if s else GREY for s in t["significant"]]
    y = np.arange(len(t))
    for yi, (_, row), c in zip(y, t.iterrows(), colors, strict=True):
        ax.plot([row["or_ci_low"], row["or_ci_high"]], [yi, yi], color=c, lw=1.5)
    ax.scatter(t["odds_ratio"], y, color=colors, zorder=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xscale("log")
    ticks = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]
    lo, hi = t["or_ci_low"].min() * 0.9, t["or_ci_high"].max() * 1.1
    ticks = [v for v in ticks if lo <= v <= hi]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in ticks]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, t.index)
    ax.set_xlabel("odds ratio (log scale), 95% CI   |   orange = p < 0.05")
    ax.set_title("Multivariable logistic regression: drivers of churn")
    _save(fig, out)


def roc_pr(y: np.ndarray, scores: dict[str, np.ndarray], out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    palette = [BLUE, ORANGE, GREEN, GREY]
    for (name, p), c in zip(scores.items(), palette, strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, color=c, label=name)
        a2.plot(rec, prec, color=c, label=name)
    a1.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    a2.axhline(y.mean(), ls="--", color=GREY, lw=1, label="random")
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve (test)")
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall curve (test)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    _save(fig, out)


def calibration(y: np.ndarray, scores: dict[str, np.ndarray], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="perfect")
    for (name, p), c in zip(scores.items(), [BLUE, ORANGE, GREEN], strict=False):
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", color=c, label=name)
    ax.set(xlabel="mean predicted probability", ylabel="observed churn rate",
           title="Calibration (reliability) curve")
    ax.legend(fontsize=8)
    _save(fig, out)


def profit_curves(curves: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, g in curves.groupby("save_rate"):
        ax.plot(g["share_contacted"], g["cumulative_value"], label=f"save rate {s:.0%}")
        best = g.loc[g["cumulative_value"].idxmax()]
        ax.scatter(best["share_contacted"], best["cumulative_value"], s=20, color="black", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set(xlabel="share of customers contacted (sorted by expected value)",
           ylabel="net value on test set ($)",
           title="Profit curve: how many customers to contact? (dot = optimum)")
    ax.legend(fontsize=8)
    _save(fig, out)


def drift_bar(drift: pd.DataFrame, warn: float, alert: float, out: Path) -> None:
    d = drift.sort_values("psi")
    colors = [ORANGE if s == "ALERT" else "#ecc94b" if s == "WARN" else BLUE for s in d["status"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["feature"], d["psi"], color=colors)
    ax.axvline(warn, color="#ecc94b", ls="--", lw=1, label=f"warn {warn}")
    ax.axvline(alert, color=ORANGE, ls="--", lw=1, label=f"alert {alert}")
    ax.set(xlabel="Population Stability Index", title="Feature drift: reference vs. new batch")
    ax.legend(fontsize=8)
    _save(fig, out)
