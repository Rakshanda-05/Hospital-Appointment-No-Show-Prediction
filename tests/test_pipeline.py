"""Unit and leakage tests for the hospital no-show prediction pipeline."""

from __future__ import annotations

import pandas as pd

from src.data_prep import clean_appointments
from src.features import NeighbourhoodEncoder, add_past_patient_history, add_static_features
from src.make_synthetic import generate_synthetic_dataset
from src.predict import load_model_bundle, predict_single


def test_synthetic_generator_schema_and_dirty_rows():
    df = generate_synthetic_dataset(n_rows=500, seed=42)
    expected_cols = {
        "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay",
        "Age", "Neighbourhood", "Scholarship", "Hipertension", "Diabetes",
        "Alcoholism", "Handcap", "SMS_received", "No-show",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == 500
    assert (df["Age"] < 0).any()
    assert (df["Age"] > 110).any()


def test_cleaning_removes_invalid_age_and_negative_lead():
    df_raw = generate_synthetic_dataset(n_rows=300, seed=42)
    df_clean, audit = clean_appointments(df_raw, data_source="synthetic")
    assert (df_clean["age"] >= 0).all()
    assert (df_clean["age"] <= 110).all()
    assert (df_clean["lead_time_days"] >= 0).all()
    assert set(df_clean["no_show"].unique()).issubset({0, 1})
    assert audit["total_rows_removed"] >= 5


def test_past_history_has_no_same_day_or_future_leakage():
    df = pd.DataFrame({
        "patient_id": [1, 1, 1, 1],
        "appointment_id": [10, 11, 12, 13],
        "appointment_date": ["2016-05-01", "2016-05-01", "2016-05-10", "2016-05-20"],
        "lead_time_days": [0, 0, 5, 10],
        "age": [30, 30, 30, 30],
        "hypertension": [0, 0, 0, 0],
        "diabetes": [0, 0, 0, 0],
        "alcoholism": [0, 0, 0, 0],
        "has_disability": [0, 0, 0, 0],
        "scholarship": [0, 0, 0, 0],
        "sms_received": [0, 0, 1, 1],
        "gender": ["F", "F", "F", "F"],
        "neighbourhood": ["CENTRO", "CENTRO", "CENTRO", "CENTRO"],
        "no_show": [1, 0, 1, 0],
    })
    out = add_static_features(add_past_patient_history(df))
    # Two visits on 2016-05-01 must both have 0 prior appointments (no same-day leakage)
    assert out.loc[0, "prior_appt_count"] == 0
    assert out.loc[1, "prior_appt_count"] == 0
    # Visit on 2016-05-10 sees both 2016-05-01 visits (1 no-show out of 2)
    assert out.loc[2, "prior_appt_count"] == 2
    assert out.loc[2, "prior_noshow_count"] == 1
    assert out.loc[2, "prior_noshow_rate"] == 0.5
    # Visit on 2016-05-20 sees all 3 earlier visits (2 no-shows out of 3)
    assert out.loc[3, "prior_appt_count"] == 3
    assert out.loc[3, "prior_noshow_count"] == 2


def test_neighbourhood_encoder_handles_unseen():
    X_tr = pd.DataFrame({"neighbourhood": ["A", "A", "B"]})
    y_tr = [1, 0, 0]
    enc = NeighbourhoodEncoder(smoothing=10.0).fit(X_tr, y_tr)
    X_te = pd.DataFrame({"neighbourhood": ["A", "UNSEEN_HOOD"]})
    transformed = enc.transform(X_te)
    assert transformed.shape == (2, 2)
    assert transformed[1, 1] == 0.0  # Unseen frequency is 0


def test_predict_single_end_to_end():
    bundle = load_model_bundle()
    res = predict_single(
        {
            "age": 24,
            "lead_time_days": 21,
            "appt_dayofweek": 1,
            "gender": "F",
            "neighbourhood": "JARDIM CAMBURI",
            "sms_received": 0,
            "scholarship": 1,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "has_disability": 0,
            "prior_appt_count": 2,
            "prior_noshow_count": 1,
        },
        bundle=bundle,
    )
    assert 0.0 <= res["no_show_probability"] <= 1.0
    assert res["risk_tier"] in {"Low", "Medium", "High"}
    assert len(res["key_drivers"]) >= 1
