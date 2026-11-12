"""Data loading, validation, and deterministic cleaning for appointment records.

Handles the real Kaggle dataset (data/raw/KaggleV2-May-2016.csv) and falls
back automatically to synthetic generation if the real file is absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.make_synthetic import generate_synthetic_dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
DEFAULT_SYNTHETIC_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_appointments.csv"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "appointments_clean.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "data_cleaning_report.json"

MAX_VALID_AGE = 110


def load_raw_data(raw_path: Path = DEFAULT_RAW_PATH) -> Tuple[pd.DataFrame, str]:
    """Load raw appointments CSV, or generate synthetic data if missing.

    Returns:
        Tuple of (raw DataFrame, data_source) where data_source is 'real' or 'synthetic'.
    """
    if raw_path.exists():
        logger.info("Loading real dataset from %s", raw_path)
        return pd.read_csv(raw_path), "real"

    logger.warning(
        "Real dataset not found at %s. Generating synthetic dataset at %s. "
        "RESULTS WILL REFLECT SYNTHETIC DATA ONLY.",
        raw_path,
        DEFAULT_SYNTHETIC_PATH,
    )
    df = generate_synthetic_dataset(output_path=DEFAULT_SYNTHETIC_PATH)
    return df, "synthetic"


def clean_appointments(df_raw: pd.DataFrame, data_source: str = "real") -> Tuple[pd.DataFrame, dict]:
    """Clean raw appointment records and return cleaned DataFrame + audit report.

    Cleaning rules:
    1. Deduplicate identical AppointmentID rows if present.
    2. Parse ScheduledDay and AppointmentDay as normalized whole-day dates
       (AppointmentDay carries no time component in the source data).
    3. Compute lead_time_days = (appointment_date - scheduled_date).dt.days.
    4. Remove rows with negative lead_time_days (AppointmentDay < ScheduledDay).
    5. Remove invalid ages (Age < 0 or Age > 110). Age == 0 (infants) is kept.
    6. Convert target 'No-show' ('Yes'/'No') to binary no_show (1/0).
    7. Standardize typo column names ('Hipertension' -> 'hypertension',
       'Handcap' -> 'handicap_raw') and create binary 'has_disability'.
    """
    df = df_raw.copy()
    audit: dict = {
        "data_source": data_source,
        "raw_rows": int(len(df)),
        "steps": [],
    }

    # 1. Deduplicate AppointmentID
    before = len(df)
    df = df.drop_duplicates(subset=["AppointmentID"]).reset_index(drop=True)
    audit["steps"].append({
        "step": "deduplicate_appointment_id",
        "rows_removed": int(before - len(df)),
        "rows_remaining": int(len(df)),
    })

    # 2. Parse dates as UTC-naive whole days
    df["scheduled_ts"] = pd.to_datetime(df["ScheduledDay"], utc=True).dt.tz_localize(None)
    df["scheduled_date"] = df["scheduled_ts"].dt.normalize()
    df["appointment_date"] = pd.to_datetime(df["AppointmentDay"], utc=True).dt.tz_localize(None).dt.normalize()
    df["lead_time_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days

    # 3. Drop negative lead times
    before = len(df)
    neg_lead_mask = df["lead_time_days"] < 0
    df = df.loc[~neg_lead_mask].copy()
    audit["steps"].append({
        "step": "drop_negative_lead_time",
        "rows_removed": int(before - len(df)),
        "rows_remaining": int(len(df)),
        "reason": "AppointmentDay strictly before ScheduledDay",
    })

    # 4. Drop invalid ages (< 0 or > MAX_VALID_AGE)
    before = len(df)
    invalid_age_mask = (df["Age"] < 0) | (df["Age"] > MAX_VALID_AGE)
    n_neg_age = int((df["Age"] < 0).sum())
    n_extreme_age = int((df["Age"] > MAX_VALID_AGE).sum())
    df = df.loc[~invalid_age_mask].copy()
    audit["steps"].append({
        "step": "drop_invalid_age",
        "rows_removed": int(before - len(df)),
        "negative_age_rows": n_neg_age,
        "extreme_age_rows_gt_110": n_extreme_age,
        "rows_remaining": int(len(df)),
        "reason": "Keep Age in [0, 110]; Age == 0 retained as valid infant visits",
    })

    # 5. Binary target: Yes -> 1 (did NOT attend), No -> 0 (attended)
    df["no_show"] = (df["No-show"].astype(str).str.strip() == "Yes").astype(int)

    # 6. Standardize columns (never expose raw PatientId in reports)
    df["patient_id"] = df["PatientId"].astype("int64")
    df["appointment_id"] = df["AppointmentID"].astype("int64")
    df["gender"] = df["Gender"].astype(str).str.strip()
    df["age"] = df["Age"].astype(int)
    df["neighbourhood"] = df["Neighbourhood"].astype(str).str.strip()
    df["scholarship"] = df["Scholarship"].astype(int)
    df["hypertension"] = df["Hipertension"].astype(int)
    df["diabetes"] = df["Diabetes"].astype(int)
    df["alcoholism"] = df["Alcoholism"].astype(int)
    df["handicap_raw"] = df["Handcap"].astype(int)
    df["has_disability"] = (df["handicap_raw"] > 0).astype(int)
    df["sms_received"] = df["SMS_received"].astype(int)

    keep_cols = [
        "patient_id",
        "appointment_id",
        "gender",
        "scheduled_date",
        "appointment_date",
        "lead_time_days",
        "age",
        "neighbourhood",
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "handicap_raw",
        "has_disability",
        "sms_received",
        "no_show",
    ]
    df_clean = df[keep_cols].sort_values(
        ["appointment_date", "scheduled_date", "appointment_id"]
    ).reset_index(drop=True)

    audit["clean_rows"] = int(len(df_clean))
    audit["total_rows_removed"] = int(audit["raw_rows"] - audit["clean_rows"])
    audit["no_show_rate"] = round(float(df_clean["no_show"].mean()), 6)
    audit["unique_patients"] = int(df_clean["patient_id"].nunique())
    audit["unique_neighbourhoods"] = int(df_clean["neighbourhood"].nunique())
    audit["appointment_date_min"] = str(df_clean["appointment_date"].min().date())
    audit["appointment_date_max"] = str(df_clean["appointment_date"].max().date())

    return df_clean, audit


def run_data_prep(
    raw_path: Path = DEFAULT_RAW_PATH,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Tuple[pd.DataFrame, dict]:
    """Load, clean, and persist processed appointments and cleaning audit JSON."""
    df_raw, data_source = load_raw_data(raw_path)
    df_clean, audit = clean_appointments(df_raw, data_source=data_source)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    return df_clean, audit


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df_out, summary = run_data_prep()
    print(json.dumps(summary, indent=2))
