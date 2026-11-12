# Data Notes & Known Pitfalls

This document records every data anomaly, schema quirk, confounding pitfall, and cleaning decision in `KaggleV2-May-2016.csv` (and the matching synthetic generator `src/make_synthetic.py`).

## 1. Source Schema & Target Encoding
- **Raw dataset:** Kaggle *"Medical Appointment No Shows"* (`data/raw/KaggleV2-May-2016.csv`), 110,527 appointments from public clinics in Vitória, Espírito Santo, Brazil (April–June 2016).
- **Target column (`No-show`):** In the raw CSV, `"Yes"` means the patient **missed** the appointment (did NOT attend), and `"No"` means the patient attended. We convert this explicitly to binary `no_show`:
  - `no_show = 1` (`"Yes"` — missed appointment, positive class, ~20.2%)
  - `no_show = 0` (`"No"` — attended appointment, ~79.8%)
- **Column-name typos preserved in raw ingestion & standardized downstream:**
  - `Hipertension` -> `hypertension`
  - `Handcap` -> `handicap_raw` + `has_disability`

## 2. Date Granularity & Lead Time
- `ScheduledDay` includes full UTC timestamps (e.g. `2016-04-29T18:38:08Z`), whereas `AppointmentDay` is recorded strictly at midnight (`00:00:00Z`) with no time-of-day component.
- Using raw timestamp subtraction would yield misleading negative fractional days whenever a same-day walk-in is scheduled after `00:00:00`.
- **Decision:** Normalize both columns to whole calendar days (`dt.tz_localize(None).dt.normalize()`) and compute `lead_time_days = (appointment_date - scheduled_date).dt.days`.
- **Invalid negative lead times:** 5 rows have `AppointmentDay` strictly before `ScheduledDay` (`lead_time_days < 0`). These are impossible scheduling records and are dropped.

## 3. Dirty Values & Outliers
- **Negative Age (`Age < 0`):** 1 row (`Age == -1`) is physically invalid and dropped.
- **Extreme Age (`Age > 110`):** 5 rows have `Age == 115` (a common legacy placeholder/typo), while 2 rows have `Age == 102`. We drop `Age > 110` (5 rows) and retain `Age == 0` (3,539 rows, valid infant appointments) and `Age <= 110`.
- **`Handcap` coded 0–4:** Levels 2, 3, and 4 have very small counts (183, 13, and 3 rows respectively out of 110,527). Treating levels 2–4 as separate ordinal or one-hot categories risks overfitting tiny cells, so we binarize `has_disability = (handicap_raw > 0).astype(int)` (2,241 positive rows) while retaining `handicap_raw` only in the cleaned audit file.
- **Duplicate `AppointmentID`:** Checked and deduplicated (0 duplicates in the real Kaggle file).
- **Row counts:** Raw `110,527` -> dropped `5` negative lead time + `6` invalid age (`1` negative, `5` > 110) = **`110,516` clean rows** (all counts logged automatically to `data/processed/data_cleaning_report.json`).

## 4. Repeat Patients & Leakage Safeguards
- **62,299 unique `PatientId`s** account for 110,527 appointments (~1.77 visits per patient on average; many patients appear multiple times).
- **Risk of ID leakage:** Never pass `PatientId` or `AppointmentID` as model features, and never print or commit `PatientId` values in reports.
- **Risk of future/same-day leakage in history features:** `prior_appt_count` and `prior_noshow_count` are computed by first aggregating daily totals per `(patient_id, appointment_date)` and taking cumulative sums strictly **before** (`<`) the current `appointment_date`. Two appointments on the same day never leak each other's outcomes.
- **Split strategy:**
  1. **Primary:** Time-based split on `AppointmentDay` (`< 2016-06-01` train vs `>= 2016-06-01` test). Returning patients visiting in June legitimately have their May history available at prediction time.
  2. **Robustness check:** 5-fold `GroupKFold` grouped by `patient_id` ensures zero patient overlap across train/validation folds.
- **High-cardinality `Neighbourhood` (81 values):** Smoothed empirical-Bayes target encoding + frequency encoding (`NeighbourhoodEncoder`) is fitted strictly inside training splits / CV training folds only.

## 5. Confounding: `SMS_received` Is Not Causal
- Unstratified, `SMS_received == 1` shows a **higher** no-show rate (~27.6%) than `SMS_received == 0` (~16.7%).
- Why? **Zero same-day appointments (`lead_time_days == 0`, n = 38,563) receive an SMS**, and same-day visits have only a ~4.6% no-show rate.
- Conditioning on advance appointments (`lead_time_days > 0`), `SMS_received == 1` actually has a slightly **lower** no-show rate (~27.6%) than `SMS_received == 0` (~29.4%).
- **Takeaway:** `SMS_received` reflects clinic operational rules (only advance appointments are eligible) plus selection effects. Never interpret model weights on `SMS_received` as causal treatment effects.

## 6. Class Imbalance (~20.2% No-Shows)
- Always predicting `"attends"` scores ~79.8% accuracy (and ~81.5% on the June test period) while catching **0%** of no-shows.
- We lead with **PR-AUC**, **ROC-AUC**, **Precision/Recall/Lift in the Top 10% and Top 20%**, and **Brier score** rather than accuracy.
