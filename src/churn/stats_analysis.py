"""Statistical exploratory analysis: which customer attributes are associated with churn?

This module answers the "why" question before any machine learning is done.
Predictive models tell us *who* is likely to churn; classical inference tells
us *which factors* are associated with churn, how strong the association is,
and how certain we are about it. Both views are needed when presenting to
business stakeholders.

Three tools are used:

1. Chi-square test of independence (categorical features)
   H0: the feature and churn are independent (churn rate is the same in every
   category). The test compares observed counts in the contingency table with
   the counts expected under independence. It is valid when expected counts are
   large enough (rule of thumb: all expected cells >= 5), which we check.
   Effect size: Cramer's V (0 = no association, 1 = perfect). A p-value only
   says whether an effect exists; with n ~ 7000 even tiny effects become
   "significant", so the effect size is what tells us whether it matters.

2. Mann-Whitney U test (numeric features)
   H0: the distribution of the feature is the same for churners and
   non-churners (more precisely, P(X_churn > X_stay) = 0.5). We use this
   rank-based test instead of a t-test because tenure and TotalCharges are
   strongly skewed and bimodal, so the normality assumption behind the t-test
   is not credible. Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2),
   which ranges from -1 to 1.

3. Multiple-testing correction (Benjamini-Hochberg)
   We run ~19 tests. At alpha = 0.05 we would expect about one false positive
   by chance alone. Benjamini-Hochberg controls the False Discovery Rate: the
   expected share of false positives among the features we call significant.

4. Multivariable logistic regression (statsmodels)
   Univariate tests can be misleading because features are correlated: e.g.
   fiber-optic customers pay more, so "high monthly charges" and "fiber optic"
   partly measure the same thing. Logistic regression estimates the effect of
   each factor *holding the others constant*. Coefficients are reported as
   odds ratios (OR): OR = 1.5 means the odds of churning are 50% higher, all
   else equal. We check multicollinearity with the Variance Inflation Factor
   (VIF); VIF > 10 means a coefficient is unstable and should not be
   interpreted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

from churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

log = logging.getLogger(__name__)


def churn_rate_ci(y: pd.Series, alpha: float = 0.05) -> tuple[float, float, float]:
    """Churn rate with a Wilson score confidence interval.

    The Wilson interval has better coverage than the textbook normal
    approximation (p +/- 1.96 * sqrt(p(1-p)/n)), especially for small groups or
    rates near 0 or 1, so we use it for every rate we report.
    """
    k, n = int(y.sum()), int(len(y))
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return k / n, float(lo), float(hi)


def cramers_v(table: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013).

    The plain formula sqrt(chi2 / (n * (min(r, c) - 1))) overestimates the
    association in finite samples; the correction subtracts the value expected
    under independence.
    """
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, c = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def categorical_tests(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_FEATURES + ["SeniorCitizen"]:
        table = pd.crosstab(df[col], df[target])
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        rates = df.groupby(col)[target].mean()
        rows.append({
            "feature": col,
            "test": "chi-square",
            "statistic": chi2,
            "dof": dof,
            "p_value": p,
            "effect_size": cramers_v(table),
            "effect_measure": "Cramer's V",
            "min_expected_count": float(expected.min()),
            "highest_churn_group": f"{rates.idxmax()} ({rates.max():.1%})",
            "lowest_churn_group": f"{rates.idxmin()} ({rates.min():.1%})",
        })
    return pd.DataFrame(rows)


def numeric_tests(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        churn = df.loc[df[target] == 1, col]
        stay = df.loc[df[target] == 0, col]
        u, p = stats.mannwhitneyu(churn, stay, alternative="two-sided")
        # Rank-biserial correlation. Positive = churners tend to have larger values.
        r_rb = 2 * u / (len(churn) * len(stay)) - 1
        rows.append({
            "feature": col,
            "test": "Mann-Whitney U",
            "statistic": u,
            "dof": np.nan,
            "p_value": p,
            "effect_size": r_rb,
            "effect_measure": "rank-biserial r",
            "min_expected_count": np.nan,
            "highest_churn_group": f"median churn={churn.median():.1f}",
            "lowest_churn_group": f"median stay={stay.median():.1f}",
        })
    return pd.DataFrame(rows)


def univariate_tests(df: pd.DataFrame, target: str, alpha: float = 0.05) -> pd.DataFrame:
    """Run all univariate tests and apply the Benjamini-Hochberg FDR correction."""
    res = pd.concat([categorical_tests(df, target), numeric_tests(df, target)], ignore_index=True)
    reject, p_adj, _, _ = multipletests(res["p_value"], alpha=alpha, method="fdr_bh")
    res["p_adj_bh"] = p_adj
    res["significant"] = reject
    res["abs_effect"] = res["effect_size"].abs()
    return res.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")


def _inference_design(df: pd.DataFrame) -> pd.DataFrame:
    """Build a design matrix suitable for *interpretation*, not just prediction.

    Choices made to avoid perfect or severe multicollinearity:
    * TotalCharges is dropped: it is approximately tenure * MonthlyCharges, so
      keeping all three makes the coefficients unstable (high VIF).
    * MonthlyCharges is dropped too. The bill is almost a deterministic sum of
      the subscribed services (fiber, streaming, extra lines, ...): with the
      service dummies in the model its VIF was ~33 and its odds ratio flipped
      to a meaningless "higher bills reduce churn". Price effects are carried
      by the service dummies instead. (The *predictive* models keep
      MonthlyCharges: multicollinearity hurts interpretation, not prediction.)
    * The add-on services have a level "No internet service" that is identical
      to InternetService == "No". We recode each add-on to 1 = has it, 0 =
      otherwise, and let the InternetService dummies capture the "no internet"
      effect.
    * Tenure is rescaled to years so its odds ratio is readable.
    * Reference levels (the omitted dummy) are the most common / baseline
      groups: Month-to-month contract, DSL, Electronic check.
    """
    X = pd.DataFrame(index=df.index)
    X["tenure_years"] = df["tenure"] / 12
    X["senior_citizen"] = df["SeniorCitizen"]
    for col in ["Partner", "Dependents", "PaperlessBilling"]:
        X[col.lower()] = (df[col] == "Yes").astype(int)
    for col in ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection",
                "StreamingTV", "StreamingMovies"]:
        X[f"has_{col}"] = (df[col] == "Yes").astype(int)
    X["multiple_lines"] = (df["MultipleLines"] == "Yes").astype(int)
    X["contract_one_year"] = (df["Contract"] == "One year").astype(int)
    X["contract_two_year"] = (df["Contract"] == "Two year").astype(int)
    X["internet_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    X["internet_none"] = (df["InternetService"] == "No").astype(int)
    X["pay_mailed_check"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    X["pay_bank_auto"] = (df["PaymentMethod"] == "Bank transfer (automatic)").astype(int)
    X["pay_card_auto"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)
    return X.astype(float)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    vif = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.DataFrame({"feature": X.columns, "VIF": vif}).sort_values("VIF", ascending=False)


def logistic_inference(df: pd.DataFrame, target: str, alpha: float = 0.05) -> dict:
    """Fit a multivariable logistic regression and return odds ratios with CIs.

    Model: logit(P(churn)) = b0 + b1*x1 + ... + bk*xk, fitted by maximum
    likelihood. exp(b_j) is the odds ratio for a one-unit increase in x_j.
    The 95% CI comes from the Wald interval b_j +/- 1.96 * SE(b_j), then
    exponentiated. If the CI excludes 1, the factor is significant at 5%.

    Goodness of fit:
    * McFadden pseudo R^2 = 1 - LL(model)/LL(intercept only). Values of
      0.2-0.4 are considered a good fit for logistic models; it is not
      comparable to OLS R^2.
    * Likelihood-ratio test vs. the intercept-only model.
    """
    X = _inference_design(df)
    y = df[target].astype(float)
    model = sm.Logit(y, sm.add_constant(X)).fit(disp=False, maxiter=200)
    conf = model.conf_int(alpha=alpha)
    out = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "odds_ratio": np.exp(model.params),
        "or_ci_low": np.exp(conf[0]),
        "or_ci_high": np.exp(conf[1]),
        "p_value": model.pvalues,
    }).drop(index="const")
    out["significant"] = out["p_value"] < alpha
    out = out.sort_values("odds_ratio", ascending=False)
    return {
        "odds_ratios": out,
        "vif": vif_table(X),
        "pseudo_r2": float(model.prsquared),
        "llr_p_value": float(model.llr_pvalue),
        "n_obs": int(model.nobs),
        "aic": float(model.aic),
    }
