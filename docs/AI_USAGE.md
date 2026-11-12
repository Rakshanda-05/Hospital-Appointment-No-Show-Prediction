# AI Assistant Usage & Human Review Log

## How an AI Coding Assistant Was Used
An AI coding assistant (Gemini) was used during development to accelerate boilerplate scaffolding, module structuring, plotting utilities, test cases, and documentation drafting under explicit engineering constraints:
- Enforcing strict leakage prevention (time-based split, strictly past-only patient history, inside-fold `Neighbourhood` encoding, `GroupKFold` validation).
- Writing a schema-compatible synthetic fallback generator (`src/make_synthetic.py`).
- Structuring the modular `src/` pipeline, Streamlit app (`app/app.py`), Jupyter notebook (`notebooks/01_eda_and_modeling.ipynb`), and documentation.

## Verification & Human Review Checklist
*(Author: fill in and check off what you personally inspected, tested, and adjusted before submission.)*
- [ ] Verified `data/raw/KaggleV2-May-2016.csv` cleaning rules and inspected `data/processed/data_cleaning_report.json`.
- [ ] Audited `src/features.py` (`add_past_patient_history` and `NeighbourhoodEncoder`) for zero future/same-day/target leakage.
- [ ] Reviewed `models/metrics.json` against the README table and confirmed no single model exceeds realistic ROC-AUC bounds without leakage.
- [ ] Inspected generated figures in `reports/figures/` and confirmed takeaways match actual plots.
- [ ] Tested `streamlit run app/app.py` interactively and verified risk tiers and responsible-AI disclaimers.
- [ ] **Changes I made / reviewed:**
  - *(Add your own notes here, e.g., threshold adjustments, feature choices, or slides/demo refinements.)*
