# 4-Minute Demo Video Script

**Target Duration:** 4:00 (240 seconds)  
**Audience:** AI/ML Internship Reviewers & Clinic Stakeholders

---

## [0:00 – 0:45] Problem Framing & Why Accuracy Fails
- **Screen:** `README.md` (Project Overview) + `docs/PLAN.md`
- **Talking Points:**
  - Public clinics face ~20% appointment no-shows, wasting clinical slots and delaying care for other patients.
  - Our goal is **not** to deny care or punish patients — it is to allocate finite outreach capacity (reminder phone calls for the top 20% highest-risk appointments, automated SMS nudges for medium risk).
  - Highlight why **accuracy is misleading**: predicting "attends" for everyone already gets ~80% accuracy with 0% recall. So we evaluate using **PR-AUC**, **ROC-AUC**, **Top-20% Lift & Recall**, and **Brier calibration**.

## [0:45 – 1:45] Data Pitfalls, Leakage Guards & Key EDA Insights
- **Screen:** `docs/DATA_NOTES.md` + `reports/figures/01_noshow_by_lead_time.png` + `reports/figures/04_sms_confounding.png`
- **Talking Points:**
  - Show cleaning audit: 110,527 raw rows -> removed 5 negative lead times and 6 invalid ages (`-1` and `115`), leaving 110,516 clean rows.
  - Show **Lead Time plot (`01`)**: same-day visits miss only ~4.7% of the time, while 15–30+ day lead times miss >32%.
  - Show **SMS Confounding plot (`04`)**: unstratified, SMS looks *correlated* with higher no-shows because same-day appointments receive zero SMS. Stratified on `lead > 0`, SMS visits actually miss slightly less (~27.6% vs ~29.4%) — textbook Simpson's Paradox.
  - Explain **3 leakage traps we prevented**:
    1. Time-based split (`< June 1` train vs `>= June 1` test) + `GroupKFold` by `PatientId`.
    2. Past-only patient history (`< AppointmentDay`, aggregating daily counts so same-day visits never leak).
    3. `Neighbourhood` target/frequency encoding fitted inside training folds only.

## [1:45 – 2:45] Modeling Results, Calibration & Error/Fairness Audit
- **Screen:** `README.md` (Results Table) + `reports/figures/07_roc_and_pr_curves.png` + `reports/figures/08_calibration_and_lift.png` + `reports/figures/10_fairness_audit.png`
- **Talking Points:**
  - Walk through the model ladder: Majority baseline -> Simple rule (`lead_time >= 14d`) -> Balanced Logistic Regression -> Tuned & Isotonic-Calibrated `HistGradientBoosting`.
  - Show Top-20% Lift & Recall: how much better the ML model concentrates true no-shows into the top 20% outreach queue compared to random calling or the simple lead-time rule.
  - Show Calibration curve (`08`) and Subgroup/Fairness audit (`09`, `10`): discuss why `Scholarship` and `Neighbourhood` are socioeconomic proxies and why human review is essential.

## [2:45 – 3:45] Live Streamlit App Demo
- **Screen:** Live browser running `streamlit run app/app.py`
- **Talking Points:**
  - Enter a **High-Risk** example: age 23, lead time 21 days, 1 prior no-show out of 2 visits, no SMS -> show predicted probability, High Risk badge, personal reminder call action, and plain-English drivers.
  - Change to a **Same-Day** visit (`lead_time_days = 0`, age 62) -> watch probability drop sharply to Low Risk.
  - Point out the Responsible AI note built directly into the UI.

## [3:45 – 4:00] Closing & Reproducibility
- **Screen:** Terminal / GitHub repo tree
- **Talking Points:**
  - Summarize reproducibility: single command `python -m src.train` runs cleaning, EDA, time-split + GroupKFold, calibration, and exports `models/metrics.json`; `pytest` verifies zero-leakage unit tests; synthetic fallback works if raw CSV is absent.
  - Thank reviewers and wrap up.
