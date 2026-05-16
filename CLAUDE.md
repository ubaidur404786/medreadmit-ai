# MedReadmit AI

## Project
End-to-end ML system predicting 30-day hospital readmission risk. Combines structured ML
(gradient boosting on tabular EHR features) with — eventually — a fine-tuned
Bio_ClinicalBERT branch on discharge notes. Late-fusion model deployed as a
FastAPI backend + Streamlit frontend with SHAP explainability, MLflow experiment
tracking, and a fairness audit across demographic groups.

## Owner
MSc Data Science & AI student (Université Côte d'Azur). Background: PyTorch,
TensorFlow, Scikit-Learn, FastAPI, Flask, Docker, MLflow, Optuna, HuggingFace,
TFLite, Knowledge Graphs. Comfortable with production code, type hints, MLflow,
hyperparameter tuning. Wants clean, typed, testable Python — not notebook spaghetti.

## Hardware constraints
- Local machine: NVIDIA RTX 3050 6 GB VRAM, 8 GB system RAM
- Local is fine for: LightGBM/XGBoost, SHAP, FastAPI, Streamlit, Docker, all inference
- Local is too tight for: ClinicalBERT fine-tuning at batch size 16 — use Google Colab
  free T4 (16 GB) for the training step, then download the fine-tuned weights
- Watch system RAM when loading the full UCI dataset — use `dtype` downcasting or
  chunked reading if needed

## Datasets
- **Primary (Modules 1 & 3):** UCI Diabetes 130-US Hospitals for Years 1999–2008
  (~101K encounters, explicit `readmitted` label with values `<30`, `>30`, `NO`).
  Fetched via the `ucimlrepo` package. No credentialing required.
  Source: https://archive.ics.uci.edu/dataset/296/
- **Stretch (Module 2 text branch):** MIMIC-IV `discharge` notes (PhysioNet
  credentialing applied for in parallel). If access doesn't arrive in time,
  substitute MT Samples or use the structured-only model for v1.

## Target variable
Binary: `readmitted_30d` = 1 if `readmitted == "<30"`, else 0.
**Drop encounters where `discharge_disposition_id == 11` (expired)** — they cannot
be readmitted, including them poisons the label.

## Architecture
- **Module 1 — Structured baseline.** LightGBM (primary), Logistic Regression
  (baseline), Optuna for tuning, SHAP for explainability, MLflow for tracking.
- **Module 2 — Text branch.** Fine-tune `emilyalsentzer/Bio_ClinicalBERT` on
  discharge notes for binary readmission. Extract [CLS] embeddings for fusion.
- **Module 3 — Fusion + product.** Late-fusion MLP combining LightGBM probability
  with the BERT [CLS] embedding. FastAPI backend (`/predict`, `/health`),
  Streamlit frontend (input form, risk gauge, SHAP plot, fairness table),
  Dockerized, deployed to Hugging Face Spaces or Render.

## Engineering conventions
- Python 3.11
- Type hints on every function signature (use `from __future__ import annotations`)
- Google-style docstrings on public functions
- `ruff` for linting, `black` for formatting (line length 100)
- `pytest` for tests — at minimum, cover `src/features/` and the FastAPI endpoints
- MLflow tracking with the local file backend (`./mlruns/`)
- Random seeds set explicitly in every training script (`random_state=42`)

## Folder structure
```
medreadmit-ai/
├── data/
│   ├── raw/            (gitignored)
│   └── processed/      (gitignored)
├── notebooks/          (EDA only — not the source of truth)
│   ├── 01_eda.ipynb
│   ├── 02_text_branch.ipynb
│   └── 03_fusion.ipynb
├── src/
│   ├── data/           (load, clean, target construction)
│   ├── features/       (build_features.py — pure, testable functions)
│   ├── models/         (train_lgbm.py, train_logreg.py, train_fusion.py)
│   ├── evaluate/       (metrics, calibration, fairness audit)
│   └── explain/        (SHAP utilities)
├── backend/            (FastAPI app)
├── frontend/           (Streamlit app)
├── models/             (saved model artifacts — gitignored if large)
├── mlruns/             (MLflow — gitignored)
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/  (CI: ruff + pytest on push)
├── requirements.txt
├── CLAUDE.md           (this file)
├── README.md
└── REPORT.md           (methodology, results, limitations, fairness findings)
```

## How to work with me (instructions for Claude Code)
1. **One file per turn.** Don't generate the whole module in one go.
2. **Explain design choices briefly** when you write code — why LightGBM here,
   why this loss, why this threshold. I want to learn, not just ship.
3. **Ask before assuming** about file paths, OS, or anything ambiguous.
4. **Flag dataset gotchas proactively.** Examples for UCI Diabetes:
   - Multiple encounters per `patient_nbr` → leakage risk; do patient-level splits
   - `discharge_disposition_id == 11` means expired — must be removed
   - Several columns are 90%+ missing (`weight`, `payer_code`, `medical_specialty`)
   - `?` is the missing-value marker, not NaN
   - `diag_1`, `diag_2`, `diag_3` are raw ICD-9 strings — need grouping
5. **Suggest improvements** if you see a better approach mid-build.
6. **Commit after each working piece** with clear messages.

## Current status
- [ ] Module 1 — Structured baseline (in progress)
- [ ] Module 2 — Text branch
- [ ] Module 3 — Fusion + deployment

_Update this section at the end of each work session._
