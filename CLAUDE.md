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
```

### MLflow UI
```powershell
mlflow ui --backend-store-uri ./mlruns
```

## Architecture

### Data flow
`src/data/download.py` → `src/data/load.py` → `src/data/make_target.py` → features → models

- **download**: fetches via `ucimlrepo` (id=296), writes `data/raw/diabetes.csv`. Skips if file exists.
- **load**: reads CSV with explicit `str` dtype for `diag_1/2/3`, strips whitespace padding throughout, converts blank strings to NaN. The raw CSV has space-padded cells — `?` is NOT the missing marker; blank strings are.
- **make_target**: drops ineligible dispositions (expired/hospice: ids `{11, 13, 14, 19, 20, 21}`), creates `readmitted_30d = int(readmitted == "<30")`, drops original `readmitted` column.

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
- [ ] Module 1 — Structured baseline (data layer complete: download, load, make_target)
- [ ] Module 2 — Text branch
- [ ] Module 3 — Fusion + deployment

_Update this section at the end of each work session._
