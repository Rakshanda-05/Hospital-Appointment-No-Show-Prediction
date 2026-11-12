"""Evaluation metrics, subgroup error/fairness analysis, and EDA/model plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid", context="notebook")


def compute_top_k_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_frac: float,
) -> Dict[str, float]:
    """Compute tie-aware Precision@K, Recall@K, and Lift@K for top k_frac riskiest rows.

    When scores tie across the K boundary (e.g., constant or binary baselines),
    fractional expectation over the tied boundary group is used so row order
    never biases the metric.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    k = max(1, int(round(k_frac * n)))
    base_rate = float(np.mean(y_true))
    total_pos = float(np.sum(y_true))

    # Sort descending by score; handle ties across the k-th slot analytically
    unique_scores = np.unique(y_score)[::-1]
    selected_pos = 0.0
    remaining = k
    for s in unique_scores:
        mask = y_score == s
        cnt = int(np.sum(mask))
        pos_in_group = float(np.sum(y_true[mask]))
        if cnt <= remaining:
            selected_pos += pos_in_group
            remaining -= cnt
            if remaining == 0:
                break
        else:
            frac = remaining / cnt
            selected_pos += frac * pos_in_group
            remaining = 0
            break

    precision_k = selected_pos / k
    recall_k = selected_pos / total_pos if total_pos > 0 else 0.0
    lift_k = precision_k / base_rate if base_rate > 0 else 0.0
    return {
        "precision": round(float(precision_k), 4),
        "recall": round(float(recall_k), 4),
        "lift": round(float(lift_k), 4),
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    threshold_label: str,
) -> Dict[str, float | str]:
    """Compute standard + operational ranking metrics for one model."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)

    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_prob)) > 1 else 0.5
    pr = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))

    top10 = compute_top_k_metrics(y_true, y_prob, 0.10)
    top20 = compute_top_k_metrics(y_true, y_prob, 0.20)

    return {
        "threshold_rule": threshold_label,
        "flagged_rate": round(float(np.mean(y_pred)), 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "brier_score": round(brier, 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy_reference_only": round(float(accuracy_score(y_true, y_pred)), 4),
        "top10_precision": top10["precision"],
        "top10_recall": top10["recall"],
        "top10_lift": top10["lift"],
        "top20_precision": top20["precision"],
        "top20_recall": top20["recall"],
        "top20_lift": top20["lift"],
    }


def compute_subgroup_metrics(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    min_count: int = 30,
) -> List[Dict[str, float | int | str]]:
    """Compute error and fairness metrics across subgroups of ``group_col``."""
    work = pd.DataFrame({
        "group": df_test[group_col].astype(str).values,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    })
    rows: List[Dict[str, float | int | str]] = []
    for grp, g in work.groupby("group", sort=True):
        n = len(g)
        if n < min_count:
            continue
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        ys = g["y_prob"].to_numpy()
        pos = int(yt.sum())
        neg = int(n - pos)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        rows.append({
            "subgroup": str(grp),
            "n": int(n),
            "base_noshow_rate": round(float(yt.mean()), 4),
            "selection_rate": round(float(yp.mean()), 4),
            "mean_pred_prob": round(float(ys.mean()), 4),
            "roc_auc": round(float(roc_auc_score(yt, ys)), 4) if len(np.unique(yt)) > 1 else None,
            "precision": round(tp / max(tp + fp, 1), 4) if (tp + fp) > 0 else 0.0,
            "recall_tpr": round(tp / max(pos, 1), 4) if pos > 0 else 0.0,
            "fnr": round(fn / max(pos, 1), 4) if pos > 0 else 0.0,
            "fpr": round(fp / max(neg, 1), 4) if neg > 0 else 0.0,
        })
    return rows


def _annotate_bars(ax: plt.Axes, rates: Iterable[float], counts: Iterable[int]) -> None:
    for patch, rate, cnt in zip(ax.patches, rates, counts):
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        ax.annotate(
            f"{rate:.1%}\n(n={cnt:,})",
            (x, y),
            ha="center",
            va="bottom",
            fontsize=8.5,
            xytext=(0, 3),
            textcoords="offset points",
        )


def generate_eda_plots(df_feat: pd.DataFrame, fig_dir: Path) -> None:
    """Generate the 6 core EDA plots with sample sizes and clear takeaways."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    base_rate = float(df_feat["no_show"].mean())

    # 1. Lead time bucket
    lead_order = ["same_day", "1_to_3_days", "4_to_7_days", "8_to_14_days", "15_to_30_days", "31_plus_days"]
    lead_labels = ["Same day\n(0d)", "1–3d", "4–7d", "8–14d", "15–30d", "31+d"]
    s1 = df_feat.groupby("lead_time_bucket")["no_show"].agg(["mean", "count"]).reindex(lead_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=lead_labels, y=s1["mean"].values, hue=lead_labels, palette="Blues_d", legend=False, ax=ax)
    ax.axhline(base_rate, color="crimson", linestyle="--", linewidth=1.2, label=f"Overall average ({base_rate:.1%})")
    _annotate_bars(ax, s1["mean"].values, s1["count"].values)
    ax.set_ylim(0, max(s1["mean"].max() * 1.25, 0.42))
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Scheduling lead time")
    ax.set_title("No-Show Rate Rises Sharply After Same-Day Appointments")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_noshow_by_lead_time.png", dpi=150)
    plt.close(fig)

    # 2. Age band
    age_order = ["0_12", "13_17", "18_29", "30_44", "45_59", "60_74", "75_plus"]
    age_labels = ["0–12", "13–17", "18–29", "30–44", "45–59", "60–74", "75+"]
    s2 = df_feat.groupby("age_band")["no_show"].agg(["mean", "count"]).reindex(age_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=age_labels, y=s2["mean"].values, hue=age_labels, palette="viridis", legend=False, ax=ax)
    ax.axhline(base_rate, color="crimson", linestyle="--", linewidth=1.2, label=f"Overall average ({base_rate:.1%})")
    _annotate_bars(ax, s2["mean"].values, s2["count"].values)
    ax.set_ylim(0, max(s2["mean"].max() * 1.25, 0.32))
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Age band (years)")
    ax.set_title("Teens & Young Adults (13–29) Have Highest No-Show Rates; Older Adults Attend More")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_noshow_by_age_band.png", dpi=150)
    plt.close(fig)

    # 3. Weekday
    day_order = [d for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] if d in df_feat["appt_weekday"].unique()]
    s3 = df_feat.groupby("appt_weekday")["no_show"].agg(["mean", "count"]).reindex(day_order)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(x=day_order, y=s3["mean"].values, hue=day_order, palette="crest", legend=False, ax=ax)
    ax.axhline(base_rate, color="crimson", linestyle="--", linewidth=1.2, label=f"Overall average ({base_rate:.1%})")
    _annotate_bars(ax, s3["mean"].values, s3["count"].values)
    ax.set_ylim(0, max(s3["mean"].max() * 1.25, 0.30))
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Appointment weekday")
    ax.set_title("Weekday Differences Are Modest (Note Tiny Saturday Sample Size)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_noshow_by_weekday.png", dpi=150)
    plt.close(fig)

    # 4. SMS confounding plot
    s4_overall = df_feat.groupby("sms_received")["no_show"].agg(["mean", "count"])
    s4_strat = (
        df_feat.assign(lead_group=np.where(df_feat["lead_time_days"] == 0, "Same-day (lead = 0d)", "Advance (lead > 0d)"))
        .groupby(["lead_group", "sms_received"])["no_show"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    # Left panel: unstratified (misleading)
    sms_lbls = ["No SMS (0)", "SMS Sent (1)"]
    sns.barplot(x=sms_lbls, y=s4_overall["mean"].values, hue=sms_lbls, palette=["#7f8c8d", "#e67e22"], legend=False, ax=axes[0])
    _annotate_bars(axes[0], s4_overall["mean"].values, s4_overall["count"].values)
    axes[0].set_title("Unstratified: SMS Looks Worse (Confounded!)")
    axes[0].set_ylabel("No-show rate")
    axes[0].set_ylim(0, 0.38)

    # Right panel: stratified by lead time > 0
    s4_strat["SMS"] = s4_strat["sms_received"].map({0: "No SMS (0)", 1: "SMS Sent (1)"})
    lead_order_s4 = ["Advance (lead > 0d)", "Same-day (lead = 0d)"]
    sms_order_s4 = ["No SMS (0)", "SMS Sent (1)"]
    sns.barplot(
        data=s4_strat, x="lead_group", y="mean", hue="SMS",
        order=lead_order_s4, hue_order=sms_order_s4,
        palette=["#7f8c8d", "#2980b9"], ax=axes[1],
    )
    s4_lookup = {(r["lead_group"], r["SMS"]): (r["mean"], int(r["count"])) for _, r in s4_strat.iterrows()}
    for hue_lbl, container in zip(sms_order_s4, axes[1].containers):
        for x_lbl, bar in zip(lead_order_s4, container):
            if (x_lbl, hue_lbl) in s4_lookup:
                val, cnt = s4_lookup[(x_lbl, hue_lbl)]
                axes[1].annotate(
                    f"{val:.1%}\n(n={cnt:,})",
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points",
                )
    axes[1].set_title("Stratified by Lead Time: Same-Day Never Gets SMS")
    axes[1].set_xlabel("")
    axes[1].legend(title="", loc="upper right")
    fig.suptitle("SMS Confounding (Simpson's Paradox): Correlation Is Not Causation", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_sms_confounding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Chronic conditions & scholarship
    cond_rows = []
    feat_order = []
    for col, label in [
        ("hypertension", "Hypertension"),
        ("diabetes", "Diabetes"),
        ("alcoholism", "Alcoholism"),
        ("has_disability", "Disability (Handcap>0)"),
        ("scholarship", "Scholarship (Bolsa Família)"),
    ]:
        feat_order.append(label)
        for val in [0, 1]:
            sub = df_feat.loc[df_feat[col] == val, "no_show"]
            cond_rows.append({
                "feature": label,
                "status": "Yes (1)" if val == 1 else "No (0)",
                "rate": float(sub.mean()),
                "count": int(len(sub)),
            })
    s5 = pd.DataFrame(cond_rows)
    status_order = ["No (0)", "Yes (1)"]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    sns.barplot(
        data=s5, x="feature", y="rate", hue="status",
        order=feat_order, hue_order=status_order,
        palette=["#95a5a6", "#2c3e50"], ax=ax,
    )
    ax.axhline(base_rate, color="crimson", linestyle="--", linewidth=1.2, label=f"Overall ({base_rate:.1%})")
    s5_lookup = {(r["feature"], r["status"]): (r["rate"], int(r["count"])) for _, r in s5.iterrows()}
    for st_lbl, container in zip(status_order, ax.containers):
        for f_lbl, patch in zip(feat_order, container):
            rate, cnt = s5_lookup[(f_lbl, st_lbl)]
            ax.annotate(
                f"{rate:.1%}\n(n={cnt:,})",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center", va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points",
            )
    ax.set_ylim(0, s5["rate"].max() * 1.38)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("")
    ax.set_title("No-Show Rates by Chronic Condition & Scholarship Status")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_noshow_by_chronic_and_scholarship.png", dpi=150)
    plt.close(fig)

    # 6. Top & bottom neighbourhoods (with n >= 200)
    hood_stats = df_feat.groupby("neighbourhood")["no_show"].agg(["mean", "count"]).reset_index()
    hood_valid = hood_stats.loc[hood_stats["count"] >= 200].sort_values("mean", ascending=False)
    top_bottom = pd.concat([hood_valid.head(8), hood_valid.tail(8)], ignore_index=True)
    top_bottom["label"] = top_bottom.apply(lambda r: f"{r['neighbourhood']} (n={int(r['count']):,})", axis=1)
    top_bottom = top_bottom.sort_values("mean", ascending=True)
    colors = ["#27ae60" if m < base_rate else "#c0392b" for m in top_bottom["mean"]]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(top_bottom["label"], top_bottom["mean"], color=colors)
    ax.axvline(base_rate, color="black", linestyle="--", linewidth=1.2, label=f"Overall ({base_rate:.1%})")
    for i, (_, row) in enumerate(top_bottom.iterrows()):
        ax.text(row["mean"] + 0.004, i, f"{row['mean']:.1%}", va="center", fontsize=8.5)
    ax.set_xlim(0, top_bottom["mean"].max() * 1.18)
    ax.set_xlabel("No-show rate")
    ax.set_title("Highest vs. Lowest No-Show Neighbourhoods (Neighbourhoods with n >= 200)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "06_noshow_by_neighbourhood.png", dpi=150)
    plt.close(fig)


def generate_model_plots(
    y_test: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    best_model_name: str,
    df_test: pd.DataFrame,
    operating_threshold: float,
    fig_dir: Path,
) -> None:
    """Generate ROC/PR, Calibration/Lift, Error Analysis, and Fairness plots."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    base_rate = float(np.mean(y_test))

    # 7. ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, probs in model_probs.items():
        uniq = np.unique(probs)
        if len(uniq) <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = roc_auc_score(y_test, probs)
        pr_val = average_precision_score(y_test, probs)
        if len(uniq) == 2:
            # Binary rule: plot exact operating point instead of misleading linear PR interpolation
            op_prec = precision_score(y_test, probs.astype(int), zero_division=0)
            op_rec = recall_score(y_test, probs.astype(int), zero_division=0)
            axes[0].plot(fpr, tpr, linestyle="--", linewidth=1.6, label=f"{name} (AUC={roc_val:.3f})")
            axes[1].scatter([op_rec], [op_prec], s=70, zorder=5, label=f"{name} (PR-AUC={pr_val:.3f})")
        else:
            axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_val:.3f})")
            prec, rec, _ = precision_recall_curve(y_test, probs)
            axes[1].plot(rec, prec, linewidth=2, label=f"{name} (PR-AUC={pr_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (0.500)")
    axes[0].set_title("ROC Curve (Time-Based Test Set)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=8.5)

    axes[1].axhline(base_rate, color="k", linestyle="--", linewidth=1, label=f"Base rate ({base_rate:.3f})")
    axes[1].set_title("Precision-Recall Curve (Time-Based Test Set)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "07_roc_and_pr_curves.png", dpi=150)
    plt.close(fig)

    # 8. Calibration curve & Decile Lift chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, probs in model_probs.items():
        if name.startswith("Baseline"):
            continue
        prob_true, prob_pred = calibration_curve(y_test, np.clip(probs, 0, 1), n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_test, np.clip(probs, 0, 1))
        axes[0].plot(prob_pred, prob_true, marker="o", linewidth=1.8, label=f"{name} (Brier={brier:.4f})")

    axes[0].plot([0, 0.7], [0, 0.7], "k--", linewidth=1, label="Perfectly calibrated")
    axes[0].set_title("Probability Calibration (10 Quantile Bins)")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Observed No-Show Fraction")
    axes[0].legend(loc="upper left", fontsize=8.5)

    best_probs = model_probs[best_model_name]
    decile_df = pd.DataFrame({"y": y_test, "p": best_probs})
    decile_df["decile"] = pd.qcut(decile_df["p"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    decile_stats = decile_df.groupby("decile", observed=False)["y"].agg(["mean", "count"]).iloc[::-1]
    decile_stats["lift"] = decile_stats["mean"] / max(base_rate, 1e-9)
    sns.barplot(x=decile_stats.index.astype(str), y=decile_stats["lift"].values, hue=decile_stats.index.astype(str), palette="YlOrRd_r", legend=False, ax=axes[1])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Base rate (Lift = 1.0x)")
    for patch, lift_val in zip(axes[1].patches, decile_stats["lift"].values):
        axes[1].annotate(f"{lift_val:.2f}x", (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                         ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    axes[1].set_title(f"Risk Decile Lift ({best_model_name})")
    axes[1].set_xlabel("Predicted Risk Decile (D10 = Highest Risk)")
    axes[1].set_ylabel("Lift over Base Rate")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "08_calibration_and_lift.png", dpi=150)
    plt.close(fig)

    # 9. Error analysis across age band and lead time bucket
    y_pred_best = (best_probs >= operating_threshold).astype(int)
    age_err = pd.DataFrame(compute_subgroup_metrics(df_test, y_test, best_probs, y_pred_best, "age_band"))
    lead_err = pd.DataFrame(compute_subgroup_metrics(df_test, y_test, best_probs, y_pred_best, "lead_time_bucket"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if not age_err.empty:
        age_order = ["0_12", "13_17", "18_29", "30_44", "45_59", "60_74", "75_plus"]
        age_err["subgroup"] = pd.Categorical(age_err["subgroup"], categories=age_order, ordered=True)
        age_err = age_err.sort_values("subgroup")
        age_m = age_err.melt(id_vars="subgroup", value_vars=["recall_tpr", "fpr", "precision"], var_name="metric", value_name="value")
        sns.barplot(data=age_m, x="subgroup", y="value", hue="metric", ax=axes[0])
        axes[0].set_title(f"Error & Recall Profile by Age Band ({best_model_name})")
        axes[0].set_xlabel("Age Band")
        axes[0].set_ylabel("Rate")
        axes[0].set_ylim(0, 1.0)
        axes[0].legend(title="")

    if not lead_err.empty:
        lead_order = ["same_day", "1_to_3_days", "4_to_7_days", "8_to_14_days", "15_to_30_days", "31_plus_days"]
        lead_err["subgroup"] = pd.Categorical(lead_err["subgroup"], categories=lead_order, ordered=True)
        lead_err = lead_err.sort_values("subgroup")
        lead_m = lead_err.melt(id_vars="subgroup", value_vars=["recall_tpr", "fpr", "precision"], var_name="metric", value_name="value")
        sns.barplot(data=lead_m, x="subgroup", y="value", hue="metric", ax=axes[1])
        axes[1].set_title(f"Error & Recall Profile by Lead Time ({best_model_name})")
        axes[1].set_xlabel("Lead Time Bucket")
        axes[1].set_ylabel("Rate")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set_ylim(0, 1.0)
        axes[1].legend(title="")

    fig.tight_layout()
    fig.savefig(fig_dir / "09_error_analysis_subgroups.png", dpi=150)
    plt.close(fig)

    # 10. Fairness audit across Gender and Scholarship
    gender_f = pd.DataFrame(compute_subgroup_metrics(df_test, y_test, best_probs, y_pred_best, "gender"))
    schol_df = df_test.assign(scholarship_lbl=df_test["scholarship"].map({0: "No Scholarship", 1: "Scholarship"}))
    schol_f = pd.DataFrame(compute_subgroup_metrics(schol_df, y_test, best_probs, y_pred_best, "scholarship_lbl"))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    if not gender_f.empty:
        gm = gender_f.melt(id_vars="subgroup", value_vars=["base_noshow_rate", "selection_rate", "recall_tpr", "fpr"], var_name="metric", value_name="value")
        sns.barplot(data=gm, x="subgroup", y="value", hue="metric", ax=axes[0])
        axes[0].set_title("Fairness Check by Gender")
        axes[0].set_xlabel("Gender")
        axes[0].set_ylabel("Rate")
        axes[0].set_ylim(0, 0.7)
        axes[0].legend(title="", fontsize=8)

    if not schol_f.empty:
        sm = schol_f.melt(id_vars="subgroup", value_vars=["base_noshow_rate", "selection_rate", "recall_tpr", "fpr"], var_name="metric", value_name="value")
        sns.barplot(data=sm, x="subgroup", y="value", hue="metric", ax=axes[1])
        axes[1].set_title("Fairness Check by Scholarship (Socioeconomic Proxy)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Rate")
        axes[1].set_ylim(0, 0.7)
        axes[1].legend(title="", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "10_fairness_audit.png", dpi=150)
    plt.close(fig)
