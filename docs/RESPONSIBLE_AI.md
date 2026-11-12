# Responsible AI, Fairness & Privacy Assessment

## 1. Intended Use & Strict Prohibition on Harm
This model (`HistGradientBoosting (isotonic cal.)`) is a **supportive outreach prioritization tool** for clinic scheduling staff.
- **Permitted use:** Allocating proactive reminder phone calls (High Risk: top 20%), automated SMS/WhatsApp reschedule prompts (Medium Risk: 60th–80th percentile), and optional transportation or social-work assistance.
- **Prohibited use:** The model **must NOT** be used to deny care, deprioritize patients on waitlists, impose no-show penalty fees, or automatically double-book appointments without clinical safety buffers.

## 2. Fairness & Subgroup Error Audit (Time-Based Test Set, n = 26,449)

### A. Gender Parity
| Gender | n | Base No-Show Rate | Flagged Rate (Selection) | ROC-AUC | Recall (TPR) | FPR | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Female (`F`) | 17,314 | 18.28% | 18.55% | 0.7344 | 36.68% | 14.49% | 36.16% |
| Male (`M`) | 9,135 | 18.80% | 17.87% | 0.7411 | 35.47% | 13.79% | 37.32% |

**Finding:** Selection rates (`18.55%` vs. `17.87%`), true-positive rates (`36.68%` vs. `35.47%`), and false-positive rates (`14.49%` vs. `13.79%`) are closely balanced across `F` and `M`.

### B. Age Band Disparities
| Age Band | n | Base No-Show Rate | Flagged Rate | ROC-AUC | Recall (TPR) | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `0_12` | 5,021 | 18.54% | 22.37% | 0.7367 | 41.46% | 58.54% |
| `13_17` | 1,477 | 25.46% | 39.40% | 0.7259 | 61.97% | 38.03% |
| `18_29` | 3,852 | 23.18% | 37.10% | 0.7339 | 60.36% | 39.64% |
| `30_44` | 5,140 | 19.67% | 21.21% | 0.7373 | 39.66% | 60.34% |
| `45_59` | 5,688 | 16.54% | 8.26% | 0.7043 | 17.22% | 82.78% |
| `60_74` | 3,790 | 13.35% | 2.77% | 0.6720 | 7.31% | 92.69% |
| `75_plus` | 1,481 | 15.12% | 2.97% | 0.6752 | 5.36% | 94.64% |

**Finding & Mitigation:** Global risk ranking concentrates reminder calls on younger patients (`13–29`), while older adults (`60+`) have very high false-negative rates (`92.7%`–`94.6%`) and lower within-group ROC-AUC (`0.672`–`0.675`). When older adults miss appointments, it is often due to unmeasured mobility, caregiver, or acute health barriers. Clinics should pair the model with a separate mobility/chronic-care check-in rule for `75+` patients rather than assuming low risk means zero support need.

### C. Socioeconomic Proxies: `Scholarship` & `Neighbourhood`
| Group | n | Base No-Show Rate | Flagged Rate | ROC-AUC | Recall (TPR) | FPR |
|---|---:|---:|---:|---:|---:|---:|
| No Scholarship (`0`) | 23,888 | 18.19% | 16.67% | 0.7314 | 32.79% | 13.08% |
| Scholarship (`1`) | 2,561 | 20.93% | 33.66% | 0.7757 | 64.37% | 25.53% |

- **`Scholarship` (Bolsa Família):** Welfare program enrollment is an explicit socioeconomic status (SES) indicator. Scholarship patients are flagged at roughly **2x the rate** (`33.66%` vs `16.67%`) with nearly 2x the false-positive rate (`25.53%` vs `13.08%`).
- **`Neighbourhood`:** Neighbourhoods correlate strongly with income, clinic transit distance, and public transport access (test base rates range from `28.22%` in `SANTOS DUMONT` to `12.73%` in `CONSOLAÇÃO` among `n >= 200` areas).
- **Why this demands supportive-only action:** Under a punitive policy, using `Scholarship` or `Neighbourhood` would systematically penalize lower-income communities. Under a **helpful outreach** policy (reminder calls, free transit vouchers, flexible rescheduling), elevated sensitivity (`64.37%` recall) directs support toward patients facing structural attendance barriers. Even so, before production deployment, clinics should test an ablation removing `Scholarship` and `Neighbourhood` to measure the fairness-vs-lift tradeoff.

## 3. Privacy & Data Governance
- The source dataset (`KaggleV2-May-2016.csv`) contains anonymized numeric `PatientId` and `AppointmentID` values with no names, addresses, phone numbers, or clinical notes.
- `PatientId` and `AppointmentID` are never passed as model features, never exported in batch prediction outputs (`src/predict.py` strips `patient_id`/`PatientId`), and never printed in reports or logs.
- Any real-world clinic deployment would require role-based access control, patient consent for automated SMS/calls, and periodic re-calibration as clinic schedules evolve.

## 4. Where Human Review Is Needed
- **Frail / older patients (`75+`)** and patients with disabilities (`has_disability == 1`), where the global model under-flags missed visits.
- **Patients with repeated consecutive no-shows**, who may need social-worker outreach or clinic-location reassignment rather than repeated automated reminders.
- **Post-intervention drift:** Once staff begin calling High-Risk patients, observed no-show rates in that tier will drop (feedback loop), requiring periodic retraining with intervention flags logged.
