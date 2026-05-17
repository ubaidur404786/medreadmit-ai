# Week 1 Results — MedReadmit AI Module 1

## Final model
Production model: `models/lgbm_calibrated.joblib` — LightGBM baseline wrapped in
`src.models.platt_wrapper.PlattWrapper` for sigmoid calibration on val.

## Test set metrics (held-out, n=14,851)
| Model                         | AUROC  | AUPRC  | Brier  | Mean pred | Mean obs |
|-------------------------------|--------|--------|--------|-----------|----------|
| LogReg baseline               | 0.6477 | 0.2016 | ~0.099 | —         | 0.1111   |
| LightGBM baseline (raw)       | 0.6621 | 0.2184 | 0.2227 | 0.4607    | 0.1111   |
| LightGBM + Optuna (50 trials) | 0.6633 | 0.2204 | —      | —         | 0.1111   |
| LightGBM + Platt scaling      | 0.6621 | 0.2184 | 0.0946 | 0.1124    | 0.1111   |

## Top SHAP features (mean |SHAP| on 5000-row test sample)
1. number_inpatient — 0.246
2. discharge_disposition_id — 0.164
3. number_diagnoses — 0.056
4. diag_1_circulatory — 0.034
5. time_in_hospital — 0.026
6. num_lab_procedures — 0.025
7. num_medications — 0.025
8. diabetesMed_No — 0.025
9. number_emergency — 0.022
10. age_[50-60) — 0.020

## Fairness findings
- Age: monotonic AUROC drop with age. <40: 0.748 (n=943). 40-65: 0.671 (n=7229). >65: 0.637 (n=6679). Calibration preserved in all groups.
- Gender: Female 0.670, Male 0.654 (within sampling noise).
- Race: large groups comparable (Caucasian 0.661, AfricanAmerican 0.656). Small groups (Hispanic, Asian, Other) too few positives for reliable AUROC.
- Admission type: range 0.656–0.676, no large disparities.

## What's already built and works
- `src/data/load.py` — load_raw() with na_values=['?']
- `src/data/make_target.py` — build_target(), drops expired/hospice rows, creates readmitted_30d
- `src/features/icd9_grouping.py` — bucket_diagnosis_columns(), 9 clinical categories
- `src/features/build_features.py` — build_features() returns (X, y, groups), 154 columns, float32, sanitized names
- `src/data/split.py` — patient_grouped_split() + assert_no_patient_leakage()
- `src/models/train_lgbm.py`, `src/models/train_logreg.py` — baselines with MLflow
- `src/models/tune_lgbm.py` — Optuna study (50 trials, TPESampler, MedianPruner)
- `src/models/calibrate_lgbm.py` — Platt scaling, fits sigmoid on val
- `src/models/platt_wrapper.py` — PlattWrapper class (the production artifact's class — must stay importable from this path)
- `src/evaluate/metrics.py` — evaluate_binary(), calibration_plot(), confusion_at_threshold()
- `src/evaluate/fairness.py` — subgroup_metrics()
- `src/explain/shap_utils.py` — TreeExplainer-based utilities
- `scripts/run_shap_analysis.py`, `scripts/run_fairness_audit.py`
- `tests/` — 11 tests passing (icd9, target, features, split)
- MLflow runs in `./mlruns/` under experiment `medreadmit-module1`

## Saved artifacts
- `models/lgbm_baseline.joblib`
- `models/lgbm_tuned.joblib`
- `models/lgbm_calibrated.joblib` ← production
- `models/logreg_baseline.joblib`
- `reports/figures/shap/` — beeswarm, importance bar, 2 waterfalls
- `reports/fairness/` — 4 subgroup CSVs
- `reports/shap_feature_importance.csv`

## Critical implementation details that must not be broken
1. **Patient-grouped split** — never use random train_test_split on this data. Always use `patient_grouped_split` from `src.data.split` with `groups=df['patient_nbr']`. 29% of encounters belong to multi-encounter patients.
2. **Feature name sanitization** — LightGBM rejects `[ ] < > , " : { } =` in feature names. The pipeline strips these. Any new feature engineering must preserve this.
3. **PlattWrapper import path** — `models/lgbm_calibrated.joblib` was pickled with `src.models.platt_wrapper.PlattWrapper`. That class must remain at that import path forever, or all calibrated artifacts break.
4. **The calibrated model wraps the baseline, not the tuned model** — tuned model saw val during retraining, so the calibration set would not be honest. baseline_lgbm trained on train only, calibrated on val, evaluated on test. This is the production choice.
5. **Discharge disposition 11 (and 13, 14, 19, 20, 21) are dropped** — expired/hospice patients can't be readmitted; including them poisons the label.

## Module 1 known limitations (for Module 2 to acknowledge, not fix)
- Single train/val split for tuning (not CV) — variance in best params
- `discharge_disposition_id` treated as numeric (it's nominal) — known modeling shortcut
- Bootstrap CIs on AUROC not computed — small subgroup metrics are point estimates only (this is one of Module 2's polish items)
- Class imbalance handled by class_weight, not resampling — calibration restores honest probabilities

## Repo state at end of Week 1
- Branch: main
- Latest commit: "Day 7: fairness audit + pytest suite + README v1 — Week 1 complete"
- GitHub: pushed to origin
- All 11 pytest tests pass
- Python 3.12, .venv at project root, `pip install -e .` already done
