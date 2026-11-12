"""Synthetic dataset generator matching the KaggleV2-May-2016.csv schema.

Used automatically as a fallback when data/raw/KaggleV2-May-2016.csv is absent,
or invoked directly via ``python -m src.make_synthetic``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42

NEIGHBOURHOODS = [
    "JARDIM CAMBURI", "MARIA ORTIZ", "RESISTENCIA", "JARDIM DA PENHA",
    "ITARARE", "CENTRO", "TABUAZEIRO", "SANTA MARTHA", "JESUS DE NAZARETH",
    "BONFIM", "SANTO ANTONIO", "SANTO ANDRE", "CARATOIRA", "JABOUR",
    "SÃO PEDRO", "ILHA DO PRINCIPE", "NOVA PALESTINA", "DA PENHA",
    "ANDORINHAS", "ROMÃO", "GURIGICA", "BELA VISTA", "PRAIA DO SUÁ",
    "MATA DA PRAIA", "ENSEADA DO SUÁ", "GRANDE VITÓRIA", "GOIABEIRAS",
    "CONSOLAÇÃO", "FORTE SÃO JOÃO", "SÃO CRISTÓVÃO",
]


def generate_synthetic_dataset(
    n_rows: int = 110_500,
    seed: int = SEED,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate a realistic synthetic hospital appointment dataset.

    Preserves the exact column names (including original typos ``Hipertension``
    and ``Handcap``), timestamp formats, repeat-patient structure, confounding
    between lead time and ``SMS_received``, ~20% no-show base rate, and a small
    set of intentional dirty rows so cleaning pipelines are exercised end to end.
    """
    rng = np.random.default_rng(seed)

    # Repeat patients (~56% unique patients relative to appointments, like real data)
    n_patients = max(1000, int(n_rows * 0.56))
    patient_pool = rng.integers(10**12, 9 * 10**14, size=n_patients, dtype=np.int64)
    patient_ages = np.clip(rng.normal(loc=38, scale=22, size=n_patients).round(), 0, 99).astype(int)
    patient_genders = rng.choice(["F", "M"], size=n_patients, p=[0.65, 0.35])
    patient_hoods = rng.choice(NEIGHBOURHOODS, size=n_patients)
    patient_scholarship = rng.binomial(1, 0.10, size=n_patients)
    patient_hypertension = np.where(patient_ages >= 40, rng.binomial(1, 0.32, n_patients), rng.binomial(1, 0.04, n_patients))
    patient_diabetes = np.where(patient_ages >= 45, rng.binomial(1, 0.14, n_patients), rng.binomial(1, 0.02, n_patients))
    patient_alcoholism = np.where(patient_ages >= 18, rng.binomial(1, 0.035, n_patients), 0)
    patient_handicap = rng.choice([0, 1, 2, 3, 4], size=n_patients, p=[0.979, 0.019, 0.0016, 0.0003, 0.0001])
    # Latent patient reliability propensity
    patient_propensity = rng.normal(0.0, 0.45, size=n_patients)

    chosen_idx = rng.integers(0, n_patients, size=n_rows)

    # Generate AppointmentDay across weekdays from 2016-04-29 to 2016-06-08
    all_days = pd.date_range("2016-04-29", "2016-06-08", freq="D")
    valid_days = all_days[all_days.dayofweek < 5]  # Mon-Fri
    appt_dates = pd.to_datetime(rng.choice(valid_days, size=n_rows))

    # Lead time: ~35% same-day (0), remainder right-skewed positive days
    is_same_day = rng.random(n_rows) < 0.35
    pos_leads = np.clip(rng.exponential(scale=14.0, size=n_rows).round().astype(int), 1, 150)
    lead_days = np.where(is_same_day, 0, pos_leads)

    sched_dates = appt_dates - pd.to_timedelta(lead_days, unit="D")
    sched_hours = rng.integers(7, 20, size=n_rows)
    sched_mins = rng.integers(0, 60, size=n_rows)
    sched_secs = rng.integers(0, 60, size=n_rows)
    sched_ts = (
        sched_dates
        + pd.to_timedelta(sched_hours, unit="h")
        + pd.to_timedelta(sched_mins, unit="m")
        + pd.to_timedelta(sched_secs, unit="s")
    )

    # SMS confounding: never sent for same-day appointments; ~50% sent for lead > 0
    sms_received = np.where(lead_days > 0, rng.binomial(1, 0.50, size=n_rows), 0)

    ages = patient_ages[chosen_idx]
    genders = patient_genders[chosen_idx]
    hoods = patient_hoods[chosen_idx]
    scholarship = patient_scholarship[chosen_idx]
    hipertension = patient_hypertension[chosen_idx]
    diabetes = patient_diabetes[chosen_idx]
    alcoholism = patient_alcoholism[chosen_idx]
    handcap = patient_handicap[chosen_idx]

    # Neighbourhood slight effect
    hood_risk_map = {h: rng.normal(0.0, 0.15) for h in NEIGHBOURHOODS}
    hood_effect = np.array([hood_risk_map[h] for h in hoods])

    # Logit for no-show calibrated around ~20% overall (~4.6% same-day, ~28.5% advance)
    young_adult = ((ages >= 15) & (ages <= 30)).astype(float)
    elderly = (ages >= 65).astype(float)
    logit = (
        -3.05
        + 1.25 * (lead_days > 0).astype(float)
        + 0.35 * np.log1p(lead_days)
        + 0.32 * young_adult
        - 0.25 * elderly
        + 0.18 * scholarship
        - 0.12 * hipertension
        + 0.20 * alcoholism
        - 0.10 * sms_received
        + hood_effect
        + patient_propensity[chosen_idx]
    )
    probs = 1.0 / (1.0 + np.exp(-logit))
    no_show_bin = rng.binomial(1, probs)
    no_show_str = np.where(no_show_bin == 1, "Yes", "No")

    appt_ids = np.arange(5_600_000, 5_600_000 + n_rows, dtype=np.int64)

    df = pd.DataFrame({
        "PatientId": patient_pool[chosen_idx].astype(float),
        "AppointmentID": appt_ids,
        "Gender": genders,
        "ScheduledDay": sched_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "AppointmentDay": appt_dates.strftime("%Y-%m-%dT00:00:00Z"),
        "Age": ages,
        "Neighbourhood": hoods,
        "Scholarship": scholarship,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "No-show": no_show_str,
    })

    # Inject a few realistic dirty rows matching known Kaggle anomalies
    if n_rows >= 20:
        df.loc[0, "Age"] = -1
        df.loc[1, "Age"] = 115
        df.loc[2, "Age"] = 115
        # AppointmentDay strictly before ScheduledDay (invalid negative lead time)
        df.loc[3, "ScheduledDay"] = "2016-05-20T10:15:00Z"
        df.loc[3, "AppointmentDay"] = "2016-05-18T00:00:00Z"
        df.loc[4, "ScheduledDay"] = "2016-06-05T14:00:00Z"
        df.loc[4, "AppointmentDay"] = "2016-06-01T00:00:00Z"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic appointment no-show CSV.")
    parser.add_argument("--n-rows", type=int, default=110_500, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_appointments.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()
    df = generate_synthetic_dataset(n_rows=args.n_rows, seed=args.seed, output_path=args.output)
    no_show_rate = (df["No-show"] == "Yes").mean()
    print(f"Generated synthetic dataset: {len(df):,} rows -> {args.output} (no-show rate: {no_show_rate:.2%})")


if __name__ == "__main__":
    main()
