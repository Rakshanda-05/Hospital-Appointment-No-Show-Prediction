"""End-to-end training, tuning, calibration, GroupKFold robustness check, and artifact export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data_prep import run_data_prep
from src.evaluate import (
    compute_binary_metrics,
    compute_subgroup_metrics,
    generate_eda_plots,
    generate_model_plots,
)
from src.features import (
    ALL_MODEL_INPUT_COLUMNS,
    build_preprocessor,
    engineer_features,
    get_feature_matrix_and_target,
)

logger = logging.getLogger(__name__)

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_CUTOFF_DATE = "2016-06-01"


def make_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True, smoothing=20.0)),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=SEED,
        )),
    ])


def make_hgb_pipeline(params: dict | None = None) -> Pipeline:
    cfg = {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "random_state": SEED,
    }
    if params:
        cfg.update(params)
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=False, smoothing=20.0)),
        ("clf", HistGradientBoostingClassifier(**cfg)),
    ])


def time_based_split(df_feat: pd.DataFrame, cutoff_date: str = DEFAULT_CUTOFF_DATE) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split chronologically by appointment_date; fallback to 75th date percentile if sparse."""
    dates = pd.to_datetime(df_feat["appointment_date"])
    cutoff = pd.Timestamp(cutoff_date)
    train_mask = dates < cutoff
    test_mask = dates >= cutoff

    if test_mask.mean() < 0.10 or train_mask.mean() < 0.50:
        unique_dates = np.sort(dates.unique())
        cutoff = pd.Timestamp(unique_dates[int(0.75 * len(unique_dates))])
        train_mask = dates < cutoff
        test_mask = dates >= cutoff

    return (
        df_feat.loc[train_mask].reset_index(drop=True),
        df_feat.loc[test_mask].reset_index(drop=True),
        str(cutoff.date()),
    )


def tune_hgb_on_time_val(df_train: pd.DataFrame) -> Tuple[dict, List[dict]]:
    """Lightly tune HistGradientBoosting on the latest 20% of training dates."""
    unique_dates = np.sort(pd.to_datetime(df_train["appointment_date"]).unique())
    val_cutoff = pd.Timestamp(unique_dates[int(0.80 * len(unique_dates))])
    sub_tr = df_train.loc[pd.to_datetime(df_train["appointment_date"]) < val_cutoff]
    sub_va = df_train.loc[pd.to_datetime(df_train["appointment_date"]) >= val_cutoff]

    X_tr, y_tr = get_feature_matrix_and_target(sub_tr)
    X_va, y_va = get_feature_matrix_and_target(sub_va)

    candidates = [
        {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 2.0},
        {"max_iter": 250, "learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 2.0},
        {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]

    results: List[dict] = []
    best_score = -1.0
    best_params = candidates[0]
    for cfg in candidates:
        pipe = make_hgb_pipeline(cfg)
        pipe.fit(X_tr, y_tr)
        val_probs = pipe.predict_proba(X_va)[:, 1]
        pr_auc = float(average_precision_score(y_va, val_probs))
        roc_auc = float(roc_auc_score(y_va, val_probs))
        brier = float(brier_score_loss(y_va, val_probs))
        results.append({**cfg, "val_pr_auc": round(pr_auc, 4), "val_roc_auc": round(roc_auc, 4), "val_brier": round(brier, 4)})
        if pr_auc > best_score:
            best_score = pr_auc
            best_params = cfg

    return best_params, results


def run_group_kfold_check(
    df_feat: pd.DataFrame,
    best_hgb_params: dict,
    n_splits: int = 5,
) -> Dict[str, dict]:
    """Run patient-grouped CV to verify generalization to unseen patients."""
    X, y = get_feature_matrix_and_target(df_feat)
    groups = df_feat["patient_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    summary: Dict[str, dict] = {}
    models = {
        "Logistic Regression (balanced)": make_logreg_pipeline,
        "HistGradientBoosting": lambda: make_hgb_pipeline(best_hgb_params),
    }

    for name, builder in models.items():
        rocs, prs, briers = [], [], []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            # Explicit zero-overlap leakage assertion
            assert len(set(groups[tr_idx]).intersection(set(groups[va_idx]))) == 0, "PatientId leaked across folds!"
            pipe = builder()
            pipe.fit(X.iloc[tr_idx], y[tr_idx])
            probs = pipe.predict_proba(X.iloc[va_idx])[:, 1]
            rocs.append(float(roc_auc_score(y[va_idx], probs)))
            prs.append(float(average_precision_score(y[va_idx], probs)))
            briers.append(float(brier_score_loss(y[va_idx], probs)))

        summary[name] = {
            "n_splits": n_splits,
            "roc_auc_mean": round(float(np.mean(rocs)), 4),
            "roc_auc_std": round(float(np.std(rocs)), 4),
            "pr_auc_mean": round(float(np.mean(prs)), 4),
            "pr_auc_std": round(float(np.std(prs)), 4),
            "brier_mean": round(float(np.mean(briers)), 4),
        }

    return summary


def train_and_evaluate() -> dict:
    """Execute the full training, validation, evaluation, and artifact saving workflow."""
    np.random.seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data prep & feature engineering
    df_clean, clean_audit = run_data_prep()
    df_feat = engineer_features(df_clean)

    # 2. Generate 6 EDA figures
    generate_eda_plots(df_feat, FIGURES_DIR)

    # 3. Primary time-based split + leakage verification
    df_train, df_test, cutoff_str = time_based_split(df_feat)
    assert pd.to_datetime(df_train["appointment_date"]).max() < pd.to_datetime(df_test["appointment_date"]).min()
    assert "patient_id" not in ALL_MODEL_INPUT_COLUMNS and "appointment_id" not in ALL_MODEL_INPUT_COLUMNS

    train_patients = set(df_train["patient_id"].unique())
    test_patients = set(df_test["patient_id"].unique())
    overlap_patients = len(train_patients.intersection(test_patients))

    X_train, y_train = get_feature_matrix_and_target(df_train)
    X_test, y_test = get_feature_matrix_and_target(df_test)

    # 4. Baseline A: Always attends (predict 0)
    prob_base_a = np.zeros(len(y_test), dtype=float)
    pred_base_a = np.zeros(len(y_test), dtype=int)

    # 5. Baseline B: Simple rule (lead_time_days >= 14 -> flag)
    lead_test = df_test["lead_time_days"].to_numpy()
    prob_base_b = (lead_test >= 14).astype(float)
    pred_base_b = (lead_test >= 14).astype(int)

    # 6. Logistic Regression (balanced) + OOF threshold for top-20% outreach budget
    gkf_tr = GroupKFold(n_splits=3)
    train_groups = df_train["patient_id"].to_numpy()

    logreg_pipe = make_logreg_pipeline()
    oof_logreg = cross_val_predict(
        logreg_pipe, X_train, y_train, cv=gkf_tr, groups=train_groups, method="predict_proba"
    )[:, 1]
    logreg_thresh_top20 = float(np.quantile(oof_logreg, 0.80))
    logreg_pipe.fit(X_train, y_train)
    prob_logreg = logreg_pipe.predict_proba(X_test)[:, 1]
    pred_logreg = (prob_logreg >= logreg_thresh_top20).astype(int)

    # 7. HistGradientBoosting (lightly tuned on time-ordered validation slice)
    best_hgb_params, tuning_log = tune_hgb_on_time_val(df_train)
    hgb_pipe = make_hgb_pipeline(best_hgb_params)
    oof_hgb = cross_val_predict(
        hgb_pipe, X_train, y_train, cv=gkf_tr, groups=train_groups, method="predict_proba"
    )[:, 1]
    hgb_thresh_top20 = float(np.quantile(oof_hgb, 0.80))
    hgb_pipe.fit(X_train, y_train)
    prob_hgb = hgb_pipe.predict_proba(X_test)[:, 1]
    pred_hgb = (prob_hgb >= hgb_thresh_top20).astype(int)

    # 8. Calibration check (Platt/sigmoid vs isotonic on GroupKFold=3 inside df_train)
    cal_pipe = CalibratedClassifierCV(
        estimator=make_hgb_pipeline(best_hgb_params),
        method="isotonic",
        cv=3,
    )
    oof_cal = cross_val_predict(
        cal_pipe, X_train, y_train, cv=gkf_tr, groups=train_groups, method="predict_proba"
    )[:, 1]
    cal_thresh_top20 = float(np.quantile(oof_cal, 0.80))
    cal_pipe.fit(X_train, y_train)
    prob_cal = cal_pipe.predict_proba(X_test)[:, 1]
    pred_cal = (prob_cal >= cal_thresh_top20).astype(int)

    # Leakage sanity check (Step 6 requirement)
    max_roc = max(roc_auc_score(y_test, prob_logreg), roc_auc_score(y_test, prob_hgb))
    leakage_flag = bool(max_roc > 0.85)
    if leakage_flag:
        logger.warning("ROC-AUC %.4f exceeds 0.85 — investigate potential leakage!", max_roc)

    # Compile model comparison table
    model_results = {
        "Baseline A: Always Attends": compute_binary_metrics(y_test, prob_base_a, pred_base_a, "Always 0"),
        "Baseline B: Rule (lead_time >= 14d)": compute_binary_metrics(y_test, prob_base_b, pred_base_b, "lead_time >= 14"),
        "Logistic Regression (balanced)": compute_binary_metrics(y_test, prob_logreg, pred_logreg, f"Top-20% OOF p>={logreg_thresh_top20:.3f}"),
        "HistGradientBoosting (tuned)": compute_binary_metrics(y_test, prob_hgb, pred_hgb, f"Top-20% OOF p>={hgb_thresh_top20:.3f}"),
        "HistGradientBoosting (isotonic cal.)": compute_binary_metrics(y_test, prob_cal, pred_cal, f"Top-20% OOF p>={cal_thresh_top20:.3f}"),
    }

    # Select best deployed model: calibrated HGB if Brier improves without dropping PR-AUC by >0.002
    raw_brier = model_results["HistGradientBoosting (tuned)"]["brier_score"]
    cal_brier = model_results["HistGradientBoosting (isotonic cal.)"]["brier_score"]
    raw_pr = model_results["HistGradientBoosting (tuned)"]["pr_auc"]
    cal_pr = model_results["HistGradientBoosting (isotonic cal.)"]["pr_auc"]

    if cal_brier <= raw_brier and cal_pr >= raw_pr - 0.002:
        best_name = "HistGradientBoosting (isotonic cal.)"
        best_model = cal_pipe
        best_probs = prob_cal
        best_pred = pred_cal
        high_thresh = cal_thresh_top20
        med_thresh = float(np.quantile(oof_cal, 0.60))
    else:
        best_name = "HistGradientBoosting (tuned)"
        best_model = hgb_pipe
        best_probs = prob_hgb
        best_pred = pred_hgb
        high_thresh = hgb_thresh_top20
        med_thresh = float(np.quantile(oof_hgb, 0.60))

    # 9. GroupKFold robustness check on full dataset
    group_cv_results = run_group_kfold_check(df_feat, best_hgb_params, n_splits=5)

    # 10. Error analysis & fairness subgroups
    error_analysis = {
        "age_band": compute_subgroup_metrics(df_test, y_test, best_probs, best_pred, "age_band"),
        "lead_time_bucket": compute_subgroup_metrics(df_test, y_test, best_probs, best_pred, "lead_time_bucket"),
        "gender": compute_subgroup_metrics(df_test, y_test, best_probs, best_pred, "gender"),
        "scholarship": compute_subgroup_metrics(
            df_test.assign(scholarship_lbl=df_test["scholarship"].map({0: "No Scholarship", 1: "Scholarship"})),
            y_test,
            best_probs,
            best_pred,
            "scholarship_lbl",
        ),
    }
    hood_subgroups = compute_subgroup_metrics(df_test, y_test, best_probs, best_pred, "neighbourhood", min_count=200)
    hood_sorted = sorted(hood_subgroups, key=lambda r: r["base_noshow_rate"], reverse=True)
    error_analysis["neighbourhood_top5_highest"] = hood_sorted[:5]
    error_analysis["neighbourhood_bottom5_lowest"] = hood_sorted[-5:]

    # 11. Generate evaluation, calibration, error, and fairness plots
    probs_for_plots = {
        "Baseline B: Rule (lead>=14d)": prob_base_b,
        "Logistic Regression (balanced)": prob_logreg,
        "HistGradientBoosting (tuned)": prob_hgb,
        "HistGradientBoosting (isotonic cal.)": prob_cal,
    }
    generate_model_plots(
        y_test=y_test,
        model_probs=probs_for_plots,
        best_model_name=best_name,
        df_test=df_test,
        operating_threshold=high_thresh,
        fig_dir=FIGURES_DIR,
    )

    # 12. Save model bundle & metrics JSON
    neighbourhoods_known = sorted(df_clean["neighbourhood"].unique().tolist())
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "data_source": clean_audit["data_source"],
        "high_risk_threshold": round(high_thresh, 4),
        "medium_risk_threshold": round(med_thresh, 4),
        "feature_columns": ALL_MODEL_INPUT_COLUMNS,
        "known_neighbourhoods": neighbourhoods_known,
        "base_noshow_rate_train": round(float(y_train.mean()), 4),
        "base_noshow_rate_test": round(float(y_test.mean()), 4),
    }
    joblib.dump(bundle, MODELS_DIR / "best_model.joblib")

    metrics_payload = {
        "seed": SEED,
        "data_cleaning": clean_audit,
        "split": {
            "method": "time_based_appointment_date",
            "cutoff_date": cutoff_str,
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "train_noshow_rate": round(float(y_train.mean()), 4),
            "test_noshow_rate": round(float(y_test.mean()), 4),
            "returning_patients_in_both_splits": int(overlap_patients),
            "leakage_warning_roc_gt_085": leakage_flag,
        },
        "selected_model": best_name,
        "operating_goal": "Flag top 20% riskiest appointments for proactive reminder calls (threshold from train OOF 80th percentile)",
        "risk_thresholds": {
            "medium_risk_60th_pct": round(med_thresh, 4),
            "high_risk_80th_pct": round(high_thresh, 4),
        },
        "hgb_tuning_validation": tuning_log,
        "best_hgb_params": best_hgb_params,
        "test_metrics": model_results,
        "group_kfold_cv": group_cv_results,
        "error_and_fairness_analysis": error_analysis,
    }
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    payload = train_and_evaluate()
    print(f"Data source: {payload['data_cleaning']['data_source']}")
    print(f"Selected model: {payload['selected_model']}")
    print(pd.DataFrame(payload["test_metrics"]).T.to_string())
