# Viva / Interview Defense Notes

Plain-English answers to likely technical and product questions from internship reviewers.

---

### 1. Why did you choose PR-AUC, ROC-AUC, and Top-20% Lift instead of Accuracy?
- In this dataset, ~80% of appointments are attended (`79.81%` overall; `81.54%` in the June test period).
- A dummy baseline ("always predict attends") scores **81.54% accuracy** while catching **0%** of no-shows.
- Clinics also have finite staff hours for reminder phone calls. **PR-AUC** (`0.3500` vs `0.1846` base rate) measures ranking quality on the minority no-show class, while **Top-20% Precision (`36.34%`), Recall (`39.38%`), and Lift (`1.97x`)** directly answer: *"If we can only call 20% of tomorrow's schedule, how many missed visits do we intercept?"*

### 2. Why use a time-based split instead of a random train/test split?
- In production, a clinic trains on past months and predicts next week's schedule. A random split leaks future clinic trends and same-week patterns across train and test.
- Splitting chronologically at `2016-06-01` (train `2016-04-29` to `2016-05-31`, `n=84,067`; test `2016-06-01` to `2016-06-08`, `n=26,449`) surfaces real temporal drift: the no-show rate shifts from `20.73%` in train to `18.46%` in test.
- We also run **5-fold `GroupKFold` by `PatientId`** (`ROC-AUC = 0.7390 ± 0.0011`, `PR-AUC = 0.3789 ± 0.0071`) to confirm the model generalizes to completely unseen patients.

### 3. How did you prevent data leakage?
1. **No IDs as features:** Dropped `PatientId` and `AppointmentID`.
2. **Strictly past-only patient history:** In `src/features.py`, we aggregate daily visits per `(patient_id, appointment_date)` first and subtract the current day's total from the cumulative sum. That guarantees same-day multiple appointments never leak each other's outcomes, and future appointments are invisible.
3. **Inside-fold Neighbourhood encoding:** `NeighbourhoodEncoder` (smoothed target + frequency encoding) is a scikit-learn `TransformerMixin` fitted strictly on training folds inside the `Pipeline`.
4. **Threshold selection on training OOF scores:** The Top-20% threshold (`p >= 0.3252`) comes from 3-fold out-of-fold predictions inside `df_train`, never from `df_test`.
5. **ROC-AUC sanity check:** Final test ROC-AUC is `0.7368` (well below `0.85`), consistent with non-leaky behavioral prediction.

### 4. What did the baselines score, and does ML actually beat a simple rule?
- **Baseline A (Always Attends):** `0.5000` ROC-AUC, `0.1846` PR-AUC, `0.0%` recall, `1.00x` Top-20% lift.
- **Baseline B (Simple Rule: `lead_time_days >= 14`):** Flags `30.31%` of test visits; scores `0.6065` ROC-AUC, `0.2350` PR-AUC, `1.57x` Top-20% lift (`31.46%` Top-20% recall).
- **HistGradientBoosting (isotonic cal.):** Scores **`0.7368` ROC-AUC**, **`0.3500` PR-AUC**, and **`1.97x` Top-20% lift** (`39.38%` Top-20% recall; **`2.16x`** Top-10% lift), while using far fewer reminder calls (`18.20%` flagged vs `30.31%`).

### 5. Why is `SMS_received` tricky in this dataset?
- Unstratified, appointments with `SMS_received == 1` appear to miss *more* often (`27.57%`) than `SMS_received == 0` (`16.7%`).
- That is **confounding (Simpson's Paradox)**: all `38,563` same-day appointments receive **zero SMS** (`SMS_received == 0`) and have only a `4.65%` no-show rate. Looking only at advance appointments (`lead_time_days > 0`), `SMS_received == 1` actually has a slightly lower no-show rate (`27.57%`) than `SMS_received == 0` (`29.44%`). So SMS does not *cause* no-shows; observational features cannot establish causal treatment effects.

### 6. Where does the model fail or struggle?
- **Short advance lead times (1–3 days) & same-day visits:** Within `1_to_3_days` (`n=4,102`), within-bucket ROC-AUC drops to `0.5795` because unexpected same-week emergencies/transit failures aren't recorded in scheduling metadata.
- **Older adults (`60–74` and `75+`):** Within-group ROC-AUC is lower (`0.6720` / `0.6752`) and recall at the global Top-20% threshold is very low (`7.31%` / `5.36%`).
- **Missing context:** No appointment specialty (e.g., psychiatry vs pediatrics), physician/clinic ID, weather, or public transit distance.

### 7. How would you improve this in a real hospital deployment?
1. Add **appointment specialty, clinic site, and scheduled time-of-day** (morning vs late afternoon).
2. Add **patient-preferred communication channel** and track whether a reminder call was actually answered (to handle intervention feedback loops).
3. Run a **randomized outreach pilot (A/B test)** on high-risk appointments to measure the causal uplift of phone calls vs SMS vs transit vouchers, rather than assuming observational `SMS_received` effects.
4. Add a **post-processing safety rule** for elderly (`75+`) or high-risk chronic patients so they still receive supportive check-ins.
