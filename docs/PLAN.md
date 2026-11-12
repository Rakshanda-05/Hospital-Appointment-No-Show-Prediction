# Project Plan: Hospital Appointment No-Show Prediction

## 1. Problem Framing
- **Domain & Users:** Public outpatient clinics (modeled on the Vitória, Espírito Santo, Brazil public health system dataset). Primary users are **clinic scheduling coordinators** and **patient outreach staff**.
- **Operational Action Triggered:** Rather than denying care or automatically double-booking patients, predictions allocate **finite outreach capacity** 48–72 hours before the appointment date:
  - **High Risk (Top 20% of risk scores):** Personal reminder phone call + optional transport/rescheduling assistance.
  - **Medium Risk (60th–80th percentile):** Automated SMS/WhatsApp confirmation prompt with an easy one-tap reschedule link.
  - **Low Risk (Bottom 60%):** Standard appointment workflow.
- **Target Definition:** Binary target `no_show`:
  - `1` = Patient **did NOT attend** (`No-show == "Yes"` in raw Kaggle schema).
  - `0` = Patient **attended** (`No-show == "No"`).

## 2. Evaluation Strategy & Metrics
- **Why Accuracy is Misleading:** Roughly ~80% of appointments are attended (`~20%` no-show rate). A trivial classifier predicting "attends" for every visit achieves ~80% accuracy while catching **0%** of no-shows.
- **Primary Metrics:**
  - **PR-AUC (Average Precision):** Primary ranking metric for imbalanced binary classification (~20% base rate).
  - **ROC-AUC:** Threshold-free discrimination metric.
  - **Top-K Operational Metrics (Top 10% and Top 20%):** Precision@K, Recall@K, and **Lift over base rate** — directly measuring how well the model prioritizes limited reminder-call slots.
  - **Thresholded Metrics (Precision, Recall, F1):** Evaluated at the **top-20% operating threshold** derived from the training/validation score distribution (not default `0.5`, and never tuned on the test set).
  - **Calibration (Brier Score & Reliability Diagram):** Ensures predicted probabilities are trustworthy enough to set risk tiers.

## 3. Validation & Leakage Prevention
- **Primary Evaluation — Time-Based Split:** Train on earlier `AppointmentDay` dates (`< 2016-06-01`, ~76% of appointments) and evaluate on the latest period (`>= 2016-06-01`, ~24% of appointments). This mirrors real deployment (predicting future appointments from past data) and surfaces temporal shift.
- **Secondary Robustness Check — Patient-Grouped CV:** 5-fold `GroupKFold` grouped by `PatientId` ensures zero overlap of patients between train and validation folds, verifying the model generalizes to unseen patients rather than memorizing repeat-patient IDs.
- **Leakage Controls:**
  1. `PatientId` and `AppointmentID` are excluded from model features.
  2. Past-only patient history (`prior_appt_count`, `prior_noshow_count`, `prior_noshow_rate`) is aggregated strictly over dates `< current AppointmentDay` so same-day or future outcomes never leak.
  3. `Neighbourhood` target/frequency encoding is fitted strictly inside the training split / CV training folds.
  4. Operating threshold (80th percentile) is computed on out-of-sample validation / training scores, never on the test set.

## 4. Baselines & Planned Models
1. **Baseline A (Majority Class):** Always predict `0` ("attends"). Establishes the ~80% accuracy trap and 0% recall.
2. **Baseline B (Simple Domain Rule):** Flag appointments with `lead_time_days >= 14` (2+ weeks ahead) as high risk. Tests whether a zero-ML scheduling heuristic already captures most signal.
3. **Model 3 (Logistic Regression):** Interpretable linear baseline in a `scikit-learn` `Pipeline` (`StandardScaler`, `OneHotEncoder`, smoothed `Neighbourhood` target encoder, `class_weight="balanced"`).
4. **Model 4 (HistGradientBoostingClassifier):** Tree-based non-linear model (scikit-learn's native LightGBM-inspired histogram GBDT — chosen over external `lightgbm`/`xgboost` to keep dependencies minimal and reproducible without compiled C++ wheels). Lightly tuned on a time-ordered validation slice of the training set, with probability calibration evaluated via Brier score.
