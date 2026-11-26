"""Command-line entry point: run the whole workflow or one stage.

    python -m churn.pipeline all        # stats -> train -> decision -> monitor
    python -m churn.pipeline stats
    python -m churn.pipeline train
    python -m churn.pipeline monitor
    python -m churn.pipeline score --input new.csv --output scored.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from churn import decision, monitoring, plots
from churn.config import load_config, resolve
from churn.data import ALL_FEATURES, clean, load_dataset, split, validate
from churn.metrics import expected_calibration_error
from churn.stats_analysis import churn_rate_ci, logistic_inference, univariate_tests
from churn.train import run_training

log = logging.getLogger("churn")


# ---------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal markdown table writer (avoids an extra dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            if v == 0 and floatfmt == ".3f":
                return "0"
            if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
                return f"{v:.2e}"
            return format(v, floatfmt)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    # itertuples keeps per-column dtypes (iterrows would upcast ints to floats)
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _p(p: float) -> str:
    return "< 1e-300" if p < 1e-300 else f"= {p:.2e}"


def _dirs(cfg: dict) -> tuple[Path, Path, Path]:
    reports, figures, models = (resolve(cfg["paths"][k]) for k in ("reports", "figures", "models"))
    for d in (reports, figures, models):
        d.mkdir(parents=True, exist_ok=True)
    return reports, figures, models


# ---------------------------------------------------------------- stages
def stage_stats(cfg: dict, df: pd.DataFrame) -> dict:
    reports, figures, _ = _dirs(cfg)
    target, alpha = cfg["data"]["target"], cfg["stats"]["alpha"]
    rate, lo, hi = churn_rate_ci(df[target])
    uni = univariate_tests(df, target, alpha)
    logit = logistic_inference(df, target, alpha)

    plots.churn_by_group(df, target, figures / "churn_by_group.png")
    plots.odds_ratio_forest(logit["odds_ratios"], figures / "odds_ratios.png")

    ors = logit["odds_ratios"].reset_index(names="feature")
    text = [
        "# Statistical analysis of churn drivers",
        "",
        f"Customers: **{len(df)}**. Churn rate: **{rate:.1%}** "
        f"(95% Wilson CI {lo:.1%} to {hi:.1%}).",
        "",
        "## 1. Univariate tests (Benjamini-Hochberg corrected)",
        "",
        "Chi-square tests for categorical features (effect size: bias-corrected Cramér's V),",
        "Mann-Whitney U for numeric features (effect size: rank-biserial r; negative = churners",
        "have lower values). Sorted by absolute effect size. Rough guide for Cramér's V:",
        "0.1 small, 0.3 medium, 0.5 large.",
        "",
        md_table(uni[["feature", "test", "statistic", "p_adj_bh", "effect_size",
                      "effect_measure", "min_expected_count", "highest_churn_group",
                      "lowest_churn_group", "significant"]]),
        "",
        "## 2. Multivariable logistic regression",
        "",
        f"n = {logit['n_obs']}, McFadden pseudo R² = **{logit['pseudo_r2']:.3f}**, "
        f"likelihood-ratio test vs. null model p {_p(logit['llr_p_value'])}, "
        f"AIC = {logit['aic']:.0f}.",
        "",
        "Odds ratio > 1 increases churn odds, < 1 decreases them, holding all other",
        "factors constant. Reference groups: month-to-month contract, DSL internet,",
        "electronic check. Tenure is per year. MonthlyCharges and TotalCharges are excluded",
        "(collinear with the service dummies and tenure, see VIF table).",
        "",
        md_table(ors[["feature", "odds_ratio", "or_ci_low", "or_ci_high", "p_value",
                      "significant"]]),
        "",
        "### Multicollinearity check (VIF)",
        "",
        "VIF > 10 would indicate unstable coefficients; VIF > 5 deserves caution.",
        "",
        md_table(logit["vif"].reset_index(drop=True), floatfmt=".2f"),
        "",
    ]
    (reports / "statistical_analysis.md").write_text("\n".join(text))
    uni.to_csv(reports / "univariate_tests.csv", index=False)
    ors.to_csv(reports / "odds_ratios.csv", index=False)
    log.info("Stats report written")
    return {"churn_rate": rate, "pseudo_r2": logit["pseudo_r2"],
            "max_vif": float(logit["vif"]["VIF"].max())}


def stage_train(cfg: dict, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    reports, figures, models = _dirs(cfg)
    target = cfg["data"]["target"]
    res = run_training(train, test, cfg, models)
    y_te = test[target].to_numpy()
    best = f"{res['best_name']} (calibrated)"
    p_best = res["predictions"][best]

    plots.roc_pr(y_te, res["predictions"], figures / "roc_pr.png")
    plots.calibration(y_te, res["predictions"], figures / "calibration.png")

    # ---- decision support
    biz = cfg["business"]
    mc = test["MonthlyCharges"].to_numpy()
    policies = decision.policy_comparison(y_te, p_best, mc, biz)
    sens = decision.sensitivity(y_te, p_best, mc, biz)
    gains = decision.gains_table(y_te, p_best)
    curves = pd.concat([decision.profit_curve(y_te, p_best, mc, biz, s)
                        for s in biz["save_rate_grid"]])
    plots.profit_curves(curves, figures / "profit_curve.png")

    # ---- reports
    cv = res["cv_table"].copy()
    cv_view = pd.DataFrame({
        "model": cv["model"],
        "ROC-AUC": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv.roc_auc_mean, cv.roc_auc_std, strict=True)],
        "PR-AUC": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv.pr_auc_mean, cv.pr_auc_std, strict=True)],
        "Brier": [f"{m:.4f} ± {s:.4f}" for m, s in zip(cv.brier_mean, cv.brier_std, strict=True)],
        "Log-loss": [f"{m:.4f} ± {s:.4f}" for m, s in zip(cv.log_loss_mean, cv.log_loss_std, strict=True)],
    })
    test_rows = []
    for name, mets in res["test_metrics"].items():
        row = {"model": name}
        for m in ("roc_auc", "pr_auc", "brier", "ece"):
            v = mets[m]
            row[m] = f"{v['estimate']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
        test_rows.append(row)
    cmp_ = res["auc_comparison"]
    text = [
        "# Model evaluation and retention decision",
        "",
        "## 1. Model selection: nested 5-fold cross-validation (training set)",
        "",
        "Mean ± standard deviation across outer folds. Hyperparameters tuned in an inner 3-fold loop.",
        "",
        md_table(cv_view),
        "",
        f"Selected: **{res['best_name']}**, parameters `{res['best_params']}`, "
        f"then {cfg['model']['calibration']} calibration.",
        "",
        "## 2. Held-out test set (bootstrap 95% CI)",
        "",
        md_table(pd.DataFrame(test_rows)),
        "",
        f"Paired bootstrap, ROC-AUC difference {res['best_name']} minus {res['other_name']}: "
        f"**{cmp_['diff']:+.4f}** (95% CI {cmp_['ci_low']:+.4f} to {cmp_['ci_high']:+.4f}, "
        f"p = {cmp_['p_value']:.3f}).",
        "",
        "## 3. Gains / lift by decile (selected model, test set)",
        "",
        md_table(gains, floatfmt=".2f"),
        "",
        "## 4. Targeting policies (test set)",
        "",
        f"Assumptions: customer value = {biz['horizon_months']} months of charges, "
        f"offer cost ${biz['offer_cost']:.0f} + contact cost ${biz['contact_cost']:.0f}, "
        f"save rate {biz['save_rate']:.0%}.",
        "",
        md_table(policies, floatfmt=".2f"),
        "",
        "## 5. Sensitivity to the save-rate assumption",
        "",
        md_table(sens.drop(columns="break_even_note"), floatfmt=".2f"),
        "",
    ]
    (reports / "model_report.md").write_text("\n".join(text))
    policies.to_csv(reports / "policy_comparison.csv", index=False)
    gains.to_csv(reports / "gains_table.csv", index=False)

    metrics = {
        "selected_model": res["best_name"],
        "best_params": res["best_params"],
        "cv": res["cv_table"].to_dict(orient="records"),
        "test": res["test_metrics"],
        "auc_comparison": cmp_,
        "policies": policies.to_dict(orient="records"),
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    res["policies"] = policies
    res["gains"] = gains
    return res


def stage_monitor(cfg: dict, test: pd.DataFrame) -> dict:
    """Compare two batches with the training reference: an unchanged one and a drifted one."""
    reports, figures, models = _dirs(cfg)
    target = cfg["data"]["target"]
    model = joblib.load(models / "model.joblib")
    meta = json.loads((models / "metadata.json").read_text())
    ref = pd.read_csv(models / "reference.csv")
    p_ref = model.predict_proba(ref[ALL_FEATURES])[:, 1]

    results = {}
    sections = ["# Monitoring report", "",
                "Reference = training data. Two scoring batches are checked:",
                "* **Batch A**: the untouched test set (expected result: no drift, no false alarm).",
                "* **Batch B**: a simulated covariate shift. The test set is resampled so that",
                "  customers in their first year are 4x over-represented (an acquisition campaign).",
                "  Rows are real, so labels stay consistent and the performance check is valid.", ""]
    for label, batch in [("A_no_drift", test), ("B_simulated_drift", monitoring.simulate_drift(test))]:
        drift = monitoring.feature_drift(ref, batch, cfg)
        p_cur = model.predict_proba(batch[ALL_FEATURES])[:, 1]
        pred_psi = monitoring.psi_numeric(p_ref, p_cur, cfg["monitoring"]["psi_bins"])
        auc = roc_auc_score(batch[target], p_cur)
        perf = {
            "live_auc": float(auc),
            "baseline_auc": meta["test_roc_auc"],
            "auc_drop": float(meta["test_roc_auc"] - auc),
            "mean_predicted": float(p_cur.mean()),
            "observed_churn": float(batch[target].mean()),
            "ece": expected_calibration_error(batch[target].to_numpy(), p_cur),
            "prediction_psi": pred_psi,
        }
        status = monitoring.overall_status(drift, perf, cfg)
        if label.startswith("B"):
            plots.drift_bar(drift, cfg["monitoring"]["psi_warn"], cfg["monitoring"]["psi_alert"],
                            figures / "drift_psi.png")
        flagged = drift[drift["status"] != "OK"]
        sections += [
            f"## Batch {label.replace('_', ' ', 1)}: overall status **{status}**", "",
            f"Prediction PSI = {pred_psi:.3f}; mean predicted churn {perf['mean_predicted']:.1%} "
            f"vs. reference {p_ref.mean():.1%}. Live AUC {auc:.3f} (baseline "
            f"{meta['test_roc_auc']:.3f}); ECE {perf['ece']:.3f}.", "",
            "Flagged features:" if len(flagged) else "No feature exceeded the PSI warning level.", "",
        ]
        if len(flagged):
            sections += [md_table(flagged[["feature", "psi", "status", "test", "p_value",
                                           "ref_mean", "cur_mean"]]), ""]
        sections += ["<details><summary>All features</summary>", "",
                     md_table(drift[["feature", "type", "psi", "status", "test", "statistic",
                                     "p_value"]]), "", "</details>", ""]
        results[label] = {"status": status, "performance": perf,
                          "flagged": flagged["feature"].tolist()}
    (reports / "monitoring_report.md").write_text("\n".join(sections))
    (reports / "monitoring.json").write_text(json.dumps(results, indent=2))
    log.info("Monitoring: %s", {k: v["status"] for k, v in results.items()})
    return results


def stage_score(cfg: dict, input_path: str, output_path: str) -> pd.DataFrame:
    """Batch scoring for production: validate, predict, attach the recommended action."""
    _, _, models = _dirs(cfg)
    model = joblib.load(models / "model.joblib")
    df = clean(pd.read_csv(input_path), target=cfg["data"]["target"])
    validate(df, require_target=False)
    biz = cfg["business"]
    p = model.predict_proba(df[ALL_FEATURES])[:, 1]
    clv = decision.customer_value(df["MonthlyCharges"], biz["horizon_months"])
    ev = decision.expected_value(p, clv, biz["save_rate"], biz["offer_cost"] + biz["contact_cost"])
    out = pd.DataFrame({
        cfg["data"]["id_column"]: df[cfg["data"]["id_column"]],
        "churn_probability": p.round(4),
        "expected_value_of_offer": ev.round(2),
        "action": np.where(ev > 0, "send_retention_offer", "no_action"),
    }).sort_values("expected_value_of_offer", ascending=False)
    out.to_csv(output_path, index=False)
    log.info("Scored %d customers, %d recommended for an offer",
             len(out), int((out["action"] == "send_retention_offer").sum()))
    return out


# ---------------------------------------------------------------- tracking
def track(cfg: dict, stats_res: dict, train_res: dict, mon_res: dict) -> None:
    """Log the run to MLflow and register the model. Never breaks the pipeline."""
    if not cfg["tracking"]["enabled"] or os.environ.get("CHURN_DISABLE_MLFLOW"):
        return
    try:
        import mlflow
        import mlflow.sklearn

        t = cfg["tracking"]
        uri = t["uri"]
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            uri = "sqlite:///" + str(resolve(uri.removeprefix("sqlite:///")))
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(t["experiment"])
        best = f"{train_res['best_name']} (calibrated)"
        with mlflow.start_run(run_name=train_res["best_name"]):
            mlflow.log_params({"model": train_res["best_name"],
                               "calibration": cfg["model"]["calibration"],
                               **{k.replace("model__", ""): v
                                  for k, v in train_res["best_params"].items()},
                               **{f"biz_{k}": v for k, v in cfg["business"].items()
                                  if not isinstance(v, list)}})
            for m, v in train_res["test_metrics"][best].items():
                mlflow.log_metric(f"test_{m}", v["estimate"])
            for row in train_res["cv_table"].to_dict(orient="records"):
                mlflow.log_metric(f"cv_{row['model']}_roc_auc", row["roc_auc_mean"])
            ev = train_res["policies"].set_index("policy").loc["Expected-value rule (EV > 0)"]
            mlflow.log_metric("test_net_value_ev_rule", ev["net_value"])
            mlflow.log_metric("stats_pseudo_r2", stats_res["pseudo_r2"])
            for k, v in mon_res.items():
                mlflow.log_metric(f"monitor_{k}_prediction_psi", v["performance"]["prediction_psi"])
            mlflow.log_artifacts(str(resolve(cfg["paths"]["reports"])), artifact_path="reports")
            # MLflow >= 3.x serialises sklearn models with skops, which refuses
            # custom classes unless they are explicitly trusted.
            import inspect

            extra = {}
            if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
                extra["skops_trusted_types"] = [
                    "churn.features.TelcoFeatures",
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.model_selection._split.StratifiedKFold",
                ]
            mlflow.sklearn.log_model(
                train_res["model"], name="model",
                registered_model_name=t["registered_model"], **extra,
                input_example=pd.read_csv(resolve(cfg["paths"]["models"]) / "reference.csv")
                [ALL_FEATURES].head(3),
            )
        log.info("Run logged to MLflow at %s", uri)
    except Exception as exc:  # tracking is optional; the model files are already saved
        log.warning("MLflow tracking skipped: %s", exc)


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Telco churn pipeline")
    parser.add_argument("stage", choices=["all", "stats", "train", "monitor", "score"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", help="CSV to score (stage=score)")
    parser.add_argument("--output", default="scored.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    np.random.seed(cfg["split"]["random_state"])

    if args.stage == "score":
        if not args.input:
            parser.error("--input is required for stage=score")
        stage_score(cfg, args.input, args.output)
        return

    df = load_dataset(cfg)
    train, test = split(df, cfg)
    if args.stage == "stats":
        stage_stats(cfg, train)
    elif args.stage == "train":
        stage_train(cfg, train, test)
    elif args.stage == "monitor":
        stage_monitor(cfg, test)
    else:
        # Inference is done on the training split only, so the test set stays untouched.
        s = stage_stats(cfg, train)
        t = stage_train(cfg, train, test)
        m = stage_monitor(cfg, test)
        track(cfg, s, t, m)


if __name__ == "__main__":
    main()
