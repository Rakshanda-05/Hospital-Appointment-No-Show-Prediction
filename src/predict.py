"""Inference helper and CLI for predicting hospital appointment no-show risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from src.features import ALL_MODEL_INPUT_COLUMNS, add_static_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model_bundle(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> Dict[str, Any]:
    """Load saved model pipeline, thresholds, and metadata."""
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {bundle_path}. Run `python -m src.train` first."
        )
    return joblib.load(bundle_path)


def _explain_drivers(row: pd.Series) -> List[str]:
    """Produce plain-English risk drivers for an appointment (directional context, not causal)."""
    drivers: List[str] = []
    lead = int(row["lead_time_days"])
    if lead == 0:
        drivers.append("Same-day appointment (lead time = 0 days, strongly associated with attendance).")
    elif lead >= 14:
        drivers.append(f"Long scheduling lead time ({lead} days ahead), a strong risk factor for forgetting or schedule conflicts.")
    elif lead >= 4:
        drivers.append(f"Moderate scheduling lead time ({lead} days ahead).")

    prior_n = int(row.get("prior_appt_count", 0))
    prior_ns = int(row.get("prior_noshow_count", 0))
    if prior_n > 0:
        rate = prior_ns / prior_n
        if rate >= 0.4:
            drivers.append(f"Prior missed appointments ({prior_ns}/{prior_n} past visits missed).")
        elif prior_ns == 0:
            drivers.append(f"Consistent attendance history (0 missed out of {prior_n} past visits).")

    age = int(row["age"])
    if 13 <= age <= 29:
        drivers.append(f"Age {age} falls in the 13–29 bracket, which shows higher historical no-show rates.")
    elif age >= 60:
        drivers.append(f"Age {age} (older adult), which historically shows higher attendance.")

    if int(row.get("scholarship", 0)) == 1:
        drivers.append("Enrolled in Bolsa Família scholarship (proxy for socioeconomic barriers such as transit/childcare).")

    if lead > 0 and int(row.get("sms_received", 0)) == 0:
        drivers.append("Advance appointment without SMS reminder logged.")

    if not drivers:
        drivers.append("Balanced profile across lead time, age, and clinical indicators.")
    return drivers


def _recommended_action(risk_tier: str) -> str:
    if risk_tier == "High":
        return (
            "Priority Outreach: Place a personal reminder call 48h prior, confirm transport/schedule, "
            "and offer easy rescheduling if needed."
        )
    if risk_tier == "Medium":
        return (
            "Automated Nudge: Send an interactive SMS/WhatsApp confirmation prompt with a one-tap reschedule option."
        )
    return "Standard Workflow: No extra outreach required beyond routine clinic confirmation."


def predict_single(record: Dict[str, Any], bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Predict no-show probability, risk tier, and outreach recommendation for one appointment."""
    if bundle is None:
        bundle = load_model_bundle()

    raw_df = pd.DataFrame([{
        "gender": str(record.get("gender", "F")),
        "age": int(record.get("age", 35)),
        "lead_time_days": int(record.get("lead_time_days", 7)),
        "appt_dayofweek": int(record.get("appt_dayofweek", 1)),
        "neighbourhood": str(record.get("neighbourhood", "JARDIM CAMBURI")),
        "scholarship": int(record.get("scholarship", 0)),
        "hypertension": int(record.get("hypertension", 0)),
        "diabetes": int(record.get("diabetes", 0)),
        "alcoholism": int(record.get("alcoholism", 0)),
        "has_disability": int(record.get("has_disability", 0)),
        "sms_received": int(record.get("sms_received", 0)),
        "prior_appt_count": int(record.get("prior_appt_count", 0)),
        "prior_noshow_count": int(record.get("prior_noshow_count", 0)),
    }])

    feat_df = add_static_features(raw_df)
    X = feat_df[ALL_MODEL_INPUT_COLUMNS]
    prob = float(bundle["model"].predict_proba(X)[0, 1])

    high_t = float(bundle["high_risk_threshold"])
    med_t = float(bundle["medium_risk_threshold"])
    if prob >= high_t:
        tier = "High"
    elif prob >= med_t:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "no_show_probability": round(prob, 4),
        "risk_tier": tier,
        "high_risk_threshold": high_t,
        "medium_risk_threshold": med_t,
        "suggested_action": _recommended_action(tier),
        "key_drivers": _explain_drivers(feat_df.iloc[0]),
        "limitations_note": (
            "Decision-support only: intended strictly for supportive outreach (reminders, rescheduling, "
            "transport help), never to deny or penalize care."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score appointment CSV with trained no-show model.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with cleaned/static columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH, help="Model joblib path.")
    args = parser.parse_args()

    bundle = load_model_bundle(args.bundle)
    df_in = pd.read_csv(args.input)
    df_feat = add_static_features(df_in)
    probs = bundle["model"].predict_proba(df_feat[ALL_MODEL_INPUT_COLUMNS])[:, 1]

    high_t = float(bundle["high_risk_threshold"])
    med_t = float(bundle["medium_risk_threshold"])
    tiers = [
        "High" if p >= high_t else ("Medium" if p >= med_t else "Low")
        for p in probs
    ]

    out = df_in.drop(columns=["patient_id", "PatientId"], errors="ignore").copy()
    out["pred_noshow_prob"] = probs.round(4)
    out["risk_tier"] = tiers
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(json.dumps({"rows_scored": len(out), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
