"""Turn churn probabilities into a retention decision: whom should we contact?

A model with AUC 0.85 is useless until someone decides what to do with the
scores. This module translates probabilities into money.

Expected value of sending an offer to customer i
------------------------------------------------
Let
  p_i   = calibrated probability that customer i churns
  s     = save rate: P(customer stays | would churn and got the offer)
  CLV_i = value of keeping the customer = MonthlyCharges_i * horizon_months
  c     = offer_cost + contact_cost (paid for every contacted customer)

Then
  EV_i = p_i * s * CLV_i - c

We contact customer i if and only if EV_i > 0, i.e. if
  p_i > c / (s * CLV_i).
So the optimal probability threshold is *different for every customer*: a
high-value customer is worth contacting at a lower churn probability. This is
why a single fixed cut-off such as 0.5 is rarely optimal, and why the
probabilities must be calibrated (EV is linear in p_i, so a biased p_i
directly biases the decision).

Limitations stated honestly
---------------------------
* s (the save rate) is an assumption. In reality it must be measured with a
  randomised experiment (A/B test: offer vs. no offer). We therefore report a
  sensitivity analysis over a grid of save rates.
* The dataset has no treatment history, so we cannot estimate *uplift* (the
  causal effect of the offer per customer). A customer with high churn risk
  might not react to offers at all ("lost cause"), while a low-risk customer
  might take the discount anyway ("sure thing"). Uplift modelling needs data
  from a past campaign with a random control group; see README "Next steps".

Evaluating a policy on the test set
-----------------------------------
On held-out customers we know who actually churned (y_i). The realised value
of contacting customer i is y_i * s * CLV_i - c: we only gain if they really
would have churned. Summing this over the contacted set gives an unbiased
estimate of the policy's value under the assumed save rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def customer_value(monthly_charges: np.ndarray, horizon_months: int) -> np.ndarray:
    return np.asarray(monthly_charges, dtype=float) * horizon_months


def expected_value(p: np.ndarray, clv: np.ndarray, save_rate: float, cost: float) -> np.ndarray:
    return np.asarray(p) * save_rate * np.asarray(clv) - cost


def realised_value(y: np.ndarray, clv: np.ndarray, save_rate: float, cost: float) -> np.ndarray:
    return np.asarray(y) * save_rate * np.asarray(clv) - cost


def policy_comparison(
    y: np.ndarray, p: np.ndarray, monthly_charges: np.ndarray, biz: dict
) -> pd.DataFrame:
    """Compare simple targeting policies on held-out data."""
    cost = biz["offer_cost"] + biz["contact_cost"]
    s = biz["save_rate"]
    clv = customer_value(monthly_charges, biz["horizon_months"])
    value = realised_value(y, clv, s, cost)
    ev = expected_value(p, clv, s, cost)
    n = len(y)

    policies = {
        "Contact nobody": np.zeros(n, dtype=bool),
        "Contact everybody": np.ones(n, dtype=bool),
        "Fixed threshold p > 0.5": p > 0.5,
        "Top 20% by churn probability": p >= np.quantile(p, 0.80),
        "Expected-value rule (EV > 0)": ev > 0,
    }
    rows = []
    for name, mask in policies.items():
        rows.append({
            "policy": name,
            "contacted": int(mask.sum()),
            "share_contacted": float(mask.mean()),
            "churners_reached": int(y[mask].sum()),
            "recall": float(y[mask].sum() / max(1, y.sum())),
            "precision": float(y[mask].mean()) if mask.any() else 0.0,
            "net_value": float(value[mask].sum()),
            "net_value_per_1000_customers": float(value[mask].sum() / n * 1000),
        })
    return pd.DataFrame(rows)


def profit_curve(
    y: np.ndarray, p: np.ndarray, monthly_charges: np.ndarray, biz: dict, save_rate: float
) -> pd.DataFrame:
    """Cumulative realised value when contacting customers in order of expected value."""
    cost = biz["offer_cost"] + biz["contact_cost"]
    clv = customer_value(monthly_charges, biz["horizon_months"])
    order = np.argsort(-expected_value(p, clv, save_rate, cost))
    cum = np.cumsum(realised_value(y, clv, save_rate, cost)[order])
    return pd.DataFrame({
        "share_contacted": np.arange(1, len(y) + 1) / len(y),
        "cumulative_value": cum,
        "save_rate": save_rate,
    })


def sensitivity(
    y: np.ndarray, p: np.ndarray, monthly_charges: np.ndarray, biz: dict
) -> pd.DataFrame:
    """How do the optimal policy and its value change with the save-rate assumption?"""
    cost = biz["offer_cost"] + biz["contact_cost"]
    clv = customer_value(monthly_charges, biz["horizon_months"])
    rows = []
    for s in biz["save_rate_grid"]:
        mask = expected_value(p, clv, s, cost) > 0
        val = realised_value(y, clv, s, cost)
        rows.append({
            "save_rate": s,
            "share_contacted": float(mask.mean()),
            "net_value_ev_rule": float(val[mask].sum()),
            "net_value_contact_all": float(val.sum()),
            "break_even_note": "offer never pays" if not mask.any() else "",
        })
    return pd.DataFrame(rows)


def gains_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift table, the standard way to present a churn model to marketing.

    Customers are sorted by score and cut into 10 equal groups. Lift = churn
    rate in the decile / overall churn rate. A lift of 2.5 in the top decile
    means contacting those customers finds 2.5x more churners than random
    selection.
    """
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = np.arange(len(df)) * n_bins // len(df) + 1
    base = df["y"].mean()
    g = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"),
                                 mean_score=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / base
    g["cumulative_recall"] = g["churners"].cumsum() / df["y"].sum()
    return g.reset_index()
