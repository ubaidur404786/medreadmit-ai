# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
End-to-end ML system predicting 30-day hospital readmission risk. Combines structured ML
(gradient boosting on tabular EHR features) with — eventually — a fine-tuned
Bio_ClinicalBERT branch on discharge notes. Late-fusion model deployed as a
FastAPI backend + Streamlit frontend with SHAP explainability, MLflow experiment
tracking, and a fairness audit across demographic groups.

## Commands

### Setup
```powershell
# Create and activate venv, then install in editable mode with all deps
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

### Lint & format
```powershell
ruff check src tests          # lint
ruff check --fix src tests    # lint + auto-fix
black src tests               # format (line length 100)
```

### Tests
```powershell
pytest                              # all tests
pytest tests/test_foo.py            # single file
pytest tests/test_foo.py::test_bar  # single test
pytest -x -q                        # fail fast, minimal output
```

### Data pipeline (run from repo root)
```powershell
python -m src.data.download        # fetch UCI dataset → data/raw/diabetes.csv
python -m src.data.load            # smoke-test load + print missing-value summary
python -m src.data.make_target     # build readmitted_30d label, print class rate
python -m src.features.build_features  # build feature matrix, print shape + memory
python -m src.data.split           # verify patient-grouped split + leakage check
python -m src.models.train_lgbm    # train LightGBM baseline, log to MLflow
```

### MLflow UI
```powershell
mlflow ui --backend-store-uri ./mlruns
```

## Architecture

### Data flow
`src/data/download.py` → `src/data/load.py` → `src/data/make_target.py` → `src/features/build_features.py` → `src/data/split.py` → `src/models/train_lgbm.py`

- **download**: fetches via `ucimlrepo` (id=296), writes `data/raw/diabetes.csv`. Skips if file exists.
- **load**: reads CSV with explicit `str` dtype for `diag_1/2/3`, strips whitespace padding throughout, converts blank strings to NaN. The raw CSV has space-padded cells — `?` is NOT the missing marker; blank strings are.
- **make_target**: drops ineligible dispositions (expired/hospice: ids `{11, 13, 14, 19, 20, 21}`), creates `readmitted_30d = int(readmitted == "<30")`, drops original `readmitted` column.
- **build_features** (`src/features/build_features.py`): drops `encounter_id`/`weight`/`payer_code`, ICD-9 bucketing, fills informative missingness (`A1Cresult`, `max_glu_serum` → `"not_measured"`), collapses rare `medical_specialty` categories (<1%) → `"other"`, one-hot encodes all object columns, casts to float32. Returns `(X, y, groups)`.
- **icd9_grouping** (`src/features/icd9_grouping.py`): maps raw ICD-9 strings to 9 clinical groups (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other) following Strack et al. 2014.
- **split** (`src/data/split.py`): two-stage `GroupShuffleSplit` (70/15/15) keyed on `patient_nbr`. `assert_no_patient_leakage()` confirms disjoint patient sets; always call it before training.
- **metrics** (`src/evaluate/metrics.py`): `evaluate_binary()` returns AUROC, AUPRC, Brier score, best-F1 threshold (swept over PR curve — more meaningful than fixed 0.5 at 11% positive rate). `calibration_plot()` produces a reliability diagram.
- **train_lgbm** (`src/models/train_lgbm.py`): runs full pipeline, early-stopping on val AUC (patience=50), logs params/metrics/artifacts to MLflow experiment `medreadmit-module1`. Saves `models/lgbm_baseline.joblib` and `models/lgbm_val_calibration.png`.

### Module plan
- **Module 1 — Structured baseline**: LightGBM (primary), Logistic Regression (baseline), Optuna tuning, SHAP explainability, MLflow tracking. Lives in `src/models/` and `src/features/`.
- **Module 2 — Text branch**: Fine-tune `emilyalsentzer/Bio_ClinicalBERT` on discharge notes. Train on Google Colab T4 (RTX 3050 6GB is too small at batch 16); download fine-tuned weights. Lives in `src/models/train_bert.py`.
- **Module 3 — Fusion + product**: Late-fusion MLP combining LightGBM probability + BERT [CLS] embedding. FastAPI (`/predict`, `/health`), Streamlit frontend, Dockerized.

### Package layout
The `src/` directory is installed as a package (`pip install -e .`), so imports use `from src.data.load import load_raw`. Modules run as `python -m src.data.load` (not `python src/data/load.py`).

## Engineering conventions
- Python 3.11; `from __future__ import annotations` on every module
- Type hints on every function signature; Google-style docstrings on public functions
- `ruff` + `black`, line length 100
- `pytest` for tests — cover `src/features/` and FastAPI endpoints at minimum
- MLflow local backend (`./mlruns/`); random seeds `random_state=42` everywhere

## Dataset gotchas (UCI Diabetes 130-US Hospitals)
- **Patient-level splits required**: multiple encounters per `patient_nbr` — row-level splits cause leakage
- **Ineligible dispositions**: ids `{11, 13, 14, 19, 20, 21}` (expired/hospice) must be dropped before labelling
- **Missing-value marker**: blank string after stripping, NOT `"?"` (the `load_raw` function handles this)
- **High-missing columns**: `weight`, `payer_code`, `medical_specialty` are 90%+ missing
- **Diagnosis columns**: `diag_1`, `diag_2`, `diag_3` are raw ICD-9 strings (e.g. `"250.00"`, `"V58"`) — must be grouped before use as features; preserve as `str` dtype on load
- **Class imbalance**: positive rate (~11% readmitted <30d) — track AUC-ROC and PR-AUC, not accuracy

## Hardware constraints
- RTX 3050 6 GB VRAM, 8 GB RAM: fine for LightGBM, SHAP, FastAPI, Streamlit, inference
- ClinicalBERT fine-tuning: use Google Colab free T4 (16 GB), download weights afterward
- Full UCI dataset: use `dtype` downcasting or chunked reading if RAM becomes an issue

## How to work with me
1. **One file per turn** — don't generate whole modules at once
2. **Explain design choices** — why LightGBM, why this loss, why this threshold
3. **Suggest improvements** proactively if you see a better approach mid-build
4. **Commit after each working piece** with clear messages

## Current status
- [x] Module 1 — Structured baseline
  - [x] Data layer: download, load, make_target
  - [x] Features: ICD-9 bucketing, one-hot encoding, float32 cast (`src/features/`)
  - [x] Split: patient-grouped 70/15/15 with leakage guard (`src/data/split.py`)
  - [x] Evaluation utilities: AUROC/AUPRC/Brier/best-F1/calibration (`src/evaluate/metrics.py`)
  - [x] LightGBM training with MLflow tracking (`src/models/train_lgbm.py`)
  - [ ] Optuna hyperparameter tuning
  - [ ] SHAP explainability
  - [ ] Fairness audit across demographic groups
- [ ] Module 2 — Text branch (Bio_ClinicalBERT on discharge notes)
- [ ] Module 3 — Fusion + deployment (FastAPI + Streamlit)

_Update this section at the end of each work session._
