# MedReadmit AI

30-day hospital readmission prediction from structured EHR data, with SHAP explainability, calibrated probabilities, and a demographic fairness audit. Built on UCI Diabetes 130-US Hospitals (101k encounters, 70k patients). Module 1 of 3.

## Results (held-out test set, n=14,851)

| Model                         | AUROC  | AUPRC  | Brier   |
|-------------------------------|--------|--------|---------|
| Logistic Regression baseline  | 0.6477 | 0.2016 | 0.099   |
| LightGBM baseline             | 0.6621 | 0.2184 | 0.223   |
| LightGBM + Optuna (50 trials) | 0.6633 | 0.2204 | —       |
| **LightGBM + Platt scaling**  | **0.6621** | **0.2184** | **0.095** |

Tuning recovered minimal additional discrimination — UCI Diabetes has a known ceiling around AUROC 0.69 with structured features (Strack 2014, Shang 2021). The production model is the calibrated LightGBM: probabilities now reflect true frequencies (mean predicted 0.112 vs observed 0.111).

## What the model learned (SHAP)

Top features, mean |SHAP| across the test set:
1. `number_inpatient` (prior admissions) — 0.246
2. `discharge_disposition_id` — 0.164
3. `number_diagnoses` — 0.056
4. `diag_1_circulatory` — 0.034
5. `time_in_hospital` — 0.026

Prior admissions dominate, matching every major readmission study (Donzé 2013, Kansagara 2011 meta-analysis).

![SHAP summary plot](reports/figures/shap/summary_beeswarm.png)

## Fairness audit

Discrimination varies by age:

| Age group | n     | AUROC  | Mean pred | Prevalence |
|-----------|-------|--------|-----------|-----------|
| <40       | 943   | 0.748  | 0.106     | 0.111     |
| 40-65     | 7229  | 0.671  | 0.104     | 0.106     |
| >65       | 6679  | 0.637  | 0.122     | 0.117     |

Calibration is preserved across all age groups (mean predicted within 0.01 of observed). Gender and large racial subgroups show comparable performance; smaller race subgroups (Hispanic, Asian, Other) have insufficient sample for reliable subgroup metrics. Full audit in `reports/fairness/`.

## Methodology (one paragraph)

The cohort is filtered to remove expired/hospice patients (n=2,423) who cannot be readmitted. Train/val/test split is at the **patient level** using `GroupShuffleSplit` to prevent the 29% of repeat-patient encounters from leaking between splits. Features include demographics, admission/discharge codes, ICD-9 diagnoses bucketed into 9 clinical categories (Strack 2014 scheme), lab indicators (A1C, glucose), medication regimen, prior utilization. Class imbalance (11.4% positive) is handled with `class_weight="balanced"` during training; post-hoc Platt scaling on the validation set restores calibrated probabilities.

## Tech stack

Python 3.11 · pandas · scikit-learn · LightGBM · Optuna · SHAP · MLflow · pytest

## Roadmap

- **Module 1 (complete):** Structured baseline, calibrated LightGBM, SHAP, fairness audit
- **Module 2 (next):** Bio_ClinicalBERT on discharge notes (MIMIC-IV)
- **Module 3:** Late-fusion model, FastAPI backend, Streamlit dashboard, Docker, CI/CD

## Reproducing

```bash
pip install -e .
python -m src.data.download
python -m src.models.train_lgbm
python -m src.models.tune_lgbm
python -m src.models.calibrate_lgbm
python scripts/run_shap_analysis.py
python scripts/run_fairness_audit.py
pytest
```


A fairness audit with 1000-iteration percentile bootstrap CIs revealed a statistically distinguishable AUROC drop with age (40-65: 0.671 [0.649, 0.692] vs >65: 0.637 [0.616, 0.659]; CIs barely overlap). Calibration is preserved across all groups (mean predicted within 0.005 of observed). Apparent advantage for patients under 40 (point estimate 0.748) is not robust — CI [0.696, 0.799] overlaps with 40-65, and the subgroup is small (n=943). No detectable disparities by gender, race (for the two large racial subgroups), or admission type. Smaller racial subgroups (Hispanic, Other, Asian; n<300) have CIs too wide to support any claim; we report them transparently rather than excluding them.