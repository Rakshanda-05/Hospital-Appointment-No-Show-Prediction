"""Leakage-free feature engineering and sklearn-compatible preprocessing.

All patient history features are computed strictly from appointments on dates
earlier than (<) the current appointment_date. High-cardinality Neighbourhood
encoding is implemented as a scikit-learn TransformerMixin so it is fitted
strictly inside training splits / CV folds.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LEAD_TIME_BINS = [-1, 0, 3, 7, 14, 30, 10_000]
LEAD_TIME_LABELS = [
    "same_day",
    "1_to_3_days",
    "4_to_7_days",
    "8_to_14_days",
    "15_to_30_days",
    "31_plus_days",
]

AGE_BINS = [-1, 12, 17, 29, 44, 59, 74, 200]
AGE_LABELS = ["0_12", "13_17", "18_29", "30_44", "45_59", "60_74", "75_plus"]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

NUMERIC_FEATURES: List[str] = [
    "lead_time_days",
    "log_lead_time",
    "is_same_day",
    "age",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "has_disability",
    "chronic_count",
    "sms_received",
    "is_saturday",
    "has_prior_appt",
    "prior_appt_count",
    "prior_noshow_count",
    "prior_noshow_rate",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "appt_weekday",
    "age_band",
    "lead_time_bucket",
]

HIGH_CARDINALITY_FEATURES: List[str] = ["neighbourhood"]

ALL_MODEL_INPUT_COLUMNS: List[str] = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + HIGH_CARDINALITY_FEATURES
)


class NeighbourhoodEncoder(BaseEstimator, TransformerMixin):
    """Smoothed empirical-Bayes target + frequency encoder for Neighbourhood.

    Fitted strictly on training fold (X, y) to prevent target leakage across
    temporal splits or cross-validation folds.
    """

    def __init__(self, smoothing: float = 20.0) -> None:
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "NeighbourhoodEncoder":
        if isinstance(X, pd.DataFrame):
            series = X["neighbourhood"].astype(str)
        else:
            series = pd.Series(np.asarray(X).ravel().astype(str))

        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(y_arr))
        total_rows = max(len(y_arr), 1)

        stats = pd.DataFrame({"hood": series.values, "target": y_arr}).groupby("hood")["target"].agg(["count", "mean"])
        smoothed = (
            (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
            / (stats["count"] + self.smoothing)
        )
        self.target_map_ = smoothed.to_dict()
        self.freq_map_ = (stats["count"] / total_rows).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            series = X["neighbourhood"].astype(str)
        else:
            series = pd.Series(np.asarray(X).ravel().astype(str))

        target_enc = series.map(self.target_map_).fillna(self.global_mean_).to_numpy(dtype=float)
        freq_enc = series.map(self.freq_map_).fillna(0.0).to_numpy(dtype=float)
        return np.column_stack([target_enc, freq_enc])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(["neighbourhood_target_enc", "neighbourhood_freq"], dtype=object)


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic appointment-level features safe for both training and inference."""
    out = df.copy()

    if "appointment_date" in out.columns:
        appt_dt = pd.to_datetime(out["appointment_date"])
        out["appt_dayofweek"] = appt_dt.dt.dayofweek.astype(int)
    elif "appt_dayofweek" not in out.columns:
        raise ValueError("DataFrame must contain 'appointment_date' or 'appt_dayofweek'.")

    out["appt_weekday"] = out["appt_dayofweek"].map(lambda d: WEEKDAY_NAMES[int(d)])
    out["is_saturday"] = (out["appt_dayofweek"] == 5).astype(int)

    out["lead_time_days"] = out["lead_time_days"].astype(int).clip(lower=0)
    out["log_lead_time"] = np.log1p(out["lead_time_days"])
    out["is_same_day"] = (out["lead_time_days"] == 0).astype(int)
    out["lead_time_bucket"] = pd.cut(
        out["lead_time_days"],
        bins=LEAD_TIME_BINS,
        labels=LEAD_TIME_LABELS,
    ).astype(str)

    out["age"] = out["age"].astype(int)
    out["age_band"] = pd.cut(
        out["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
    ).astype(str)

    out["chronic_count"] = (
        out["hypertension"].astype(int)
        + out["diabetes"].astype(int)
        + out["alcoholism"].astype(int)
    )

    if "prior_appt_count" not in out.columns:
        out["prior_appt_count"] = 0
    if "prior_noshow_count" not in out.columns:
        out["prior_noshow_count"] = 0

    out["has_prior_appt"] = (out["prior_appt_count"] > 0).astype(int)
    out["prior_noshow_rate"] = np.where(
        out["prior_appt_count"] > 0,
        out["prior_noshow_count"] / np.maximum(out["prior_appt_count"], 1),
        0.0,
    )
    return out


def add_past_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly past-only patient appointment and no-show history.

    Aggregates daily counts per (patient_id, appointment_date) first and shifts
    cumulative sums by one day per patient so same-day multiple appointments
    never leak outcomes into one another.
    """
    out = df.copy()
    out["appointment_date"] = pd.to_datetime(out["appointment_date"])

    daily = (
        out.groupby(["patient_id", "appointment_date"], sort=True, as_index=False)
        .agg(daily_appts=("no_show", "size"), daily_noshows=("no_show", "sum"))
        .sort_values(["patient_id", "appointment_date"])
    )

    grouped = daily.groupby("patient_id", sort=False)
    daily["prior_appt_count"] = grouped["daily_appts"].cumsum() - daily["daily_appts"]
    daily["prior_noshow_count"] = grouped["daily_noshows"].cumsum() - daily["daily_noshows"]

    out = out.drop(columns=["prior_appt_count", "prior_noshow_count"], errors="ignore")
    out = out.merge(
        daily[["patient_id", "appointment_date", "prior_appt_count", "prior_noshow_count"]],
        on=["patient_id", "appointment_date"],
        how="left",
    )
    out["has_prior_appt"] = (out["prior_appt_count"] > 0).astype(int)
    out["prior_noshow_rate"] = np.where(
        out["prior_appt_count"] > 0,
        out["prior_noshow_count"] / np.maximum(out["prior_appt_count"], 1),
        0.0,
    )
    return out


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Run full deterministic feature pipeline on cleaned historical appointments."""
    df_hist = add_past_patient_history(df_clean)
    return add_static_features(df_hist)


def build_preprocessor(scale_numeric: bool = True, smoothing: float = 20.0) -> ColumnTransformer:
    """Build a leakage-safe sklearn ColumnTransformer for model pipelines."""
    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    hood_transformer = NeighbourhoodEncoder(smoothing=smoothing)

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERIC_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
            ("hood", hood_transformer, HIGH_CARDINALITY_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_matrix_and_target(df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract model input DataFrame and binary target array."""
    X = df_feat[ALL_MODEL_INPUT_COLUMNS].copy()
    y = df_feat["no_show"].to_numpy(dtype=int)
    return X, y
