"""Streamlit demo app for Hospital Appointment No-Show Risk Scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle, predict_single

METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
WEEKDAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
}


@st.cache_resource
def get_bundle():
    return load_model_bundle()


@st.cache_data
def get_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> None:
    st.set_page_config(
        page_title="Hospital No-Show Risk Assistant",
        page_icon="🏥",
        layout="wide",
    )

    st.title("Hospital Appointment No-Show Risk Assistant")
    st.caption(
        "Outreach prioritization tool for clinic scheduling coordinators — "
        "supports proactive reminders and rescheduling, never care denial."
    )

    bundle = get_bundle()
    metrics = get_metrics()

    if bundle.get("data_source") == "synthetic":
        st.warning("Running on synthetic fallback data (data/raw/KaggleV2-May-2016.csv not found).")

    col_in, col_out = st.columns([1.15, 1.0], gap="large")

    with col_in:
        st.subheader("Appointment Details")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Patient Age (years)", min_value=0, max_value=110, value=28)
            lead_time_days = st.number_input("Lead Time (days scheduled ahead)", min_value=0, max_value=180, value=14)
            weekday_name = st.selectbox("Appointment Weekday", list(WEEKDAY_MAP.keys()), index=1)
            gender = st.radio("Gender", ["F", "M"], horizontal=True)
        with c2:
            hoods = bundle.get("known_neighbourhoods", ["JARDIM CAMBURI"])
            neighbourhood = st.selectbox("Neighbourhood", hoods, index=0)
            sms_received = st.checkbox("SMS Reminder Sent", value=False)
            scholarship = st.checkbox("Bolsa Família Scholarship", value=False)
            prior_appt_count = st.number_input("Past Appointments (strictly prior)", min_value=0, max_value=50, value=0)
            prior_noshow_count = st.number_input(
                "Past No-Shows", min_value=0, max_value=int(prior_appt_count), value=0
            )

        st.markdown("**Clinical Indicators**")
        m1, m2, m3, m4 = st.columns(4)
        hypertension = m1.checkbox("Hypertension", value=False)
        diabetes = m2.checkbox("Diabetes", value=False)
        alcoholism = m3.checkbox("Alcoholism", value=False)
        has_disability = m4.checkbox("Disability", value=False)

    payload = {
        "age": age,
        "lead_time_days": lead_time_days,
        "appt_dayofweek": WEEKDAY_MAP[weekday_name],
        "gender": gender,
        "neighbourhood": neighbourhood,
        "sms_received": int(sms_received),
        "scholarship": int(scholarship),
        "hypertension": int(hypertension),
        "diabetes": int(diabetes),
        "alcoholism": int(alcoholism),
        "has_disability": int(has_disability),
        "prior_appt_count": int(prior_appt_count),
        "prior_noshow_count": int(prior_noshow_count),
    }

    result = predict_single(payload, bundle=bundle)

    with col_out:
        st.subheader("Risk Assessment & Recommended Action")
        prob_pct = result["no_show_probability"] * 100
        tier = result["risk_tier"]

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Predicted No-Show Probability", f"{prob_pct:.1f}%")
        m_col2.metric("Risk Tier", tier)

        if tier == "High":
            st.error(f"**High Risk** (score >= {result['high_risk_threshold']:.1%}) — {result['suggested_action']}")
        elif tier == "Medium":
            st.warning(f"**Medium Risk** (score >= {result['medium_risk_threshold']:.1%}) — {result['suggested_action']}")
        else:
            st.success(f"**Low Risk** — {result['suggested_action']}")

        st.markdown("**Key Contextual Drivers**")
        for d in result["key_drivers"]:
            st.markdown(f"- {d}")

        st.info(
            "**Responsible Use & Limitations:**\n"
            "- Built for **supportive outreach** (reminder calls, easy rescheduling, transit help), "
            "never to restrict or penalize access to care.\n"
            "- `SMS_received` is confounded with lead time in observational data (same-day visits receive no SMS); "
            "do not treat SMS toggles as a causal intervention estimate.\n"
            "- Predictions reflect historical 2016 Vitória clinic patterns and lack visit specialty or transport data."
        )

    if metrics:
        with st.expander("Model Evaluation Snapshot (Time-Based Test Set)"):
            sel = metrics["selected_model"]
            sel_m = metrics["test_metrics"][sel]
            st.write(
                f"**Model:** `{sel}` | **Data source:** `{metrics['data_cleaning']['data_source']}` | "
                f"**ROC-AUC:** `{sel_m['roc_auc']}` | **PR-AUC:** `{sel_m['pr_auc']}` | "
                f"**Top-20% Lift:** `{sel_m['top20_lift']}x` | **Top-20% Recall:** `{sel_m['top20_recall']:.1%}`"
            )


if __name__ == "__main__":
    main()
