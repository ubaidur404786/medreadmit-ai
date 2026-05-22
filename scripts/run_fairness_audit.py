"""Demographic and clinical fairness audit for the calibrated readmission model.

Evaluates subgroup AUROC, AUPRC, prevalence, and mean predicted risk across
age group, gender, race, and admission type on the held-out test set.

Run from the repo root after calibrate_lgbm.py has completed:
    python scripts/run_fairness_audit.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import patient_grouped_split
from src.evaluate.fairness import format_fairness_table, subgroup_metrics
from src.features.build_features import build_features

logger = logging.getLogger(__name__)

_CALIBRATED_PATH = Path("models/lgbm_calibrated.joblib")
_FAIRNESS_DIR = Path("reports/fairness")

# Age ranges as printed in the UCI dataset → coarse clinical bracket.
# [60-70) spans 60-69; we assign it to 40-65 because the decade majority
# (60-64) falls below 65 and splitting a decade bin is not possible.
_AGE_BUCKET: dict[str, str] = {
    "[0-10)": "<40",
    "[10-20)": "<40",
    "[20-30)": "<40",
    "[30-40)": "<40",
    "[40-50)": "40-65",
    "[50-60)": "40-65",
    "[60-70)": "40-65",
    "[70-80)": ">65",
    "[80-90)": ">65",
    "[90-100)": ">65",
}

# admission_type_id codes from the UCI data dictionary.
_ADMISSION_TYPE: dict[int, str] = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
}


def get_raw_test_rows(df_raw: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Return the original pre-encoding rows that correspond to the test split.

    ``build_features`` calls ``reset_index(drop=True)`` internally, so the
    integer values in ``X_test.index`` are positional offsets into that
    reset-indexed frame.  Re-applying ``reset_index`` to ``df_raw`` before
    ``.iloc`` aligns the two indices.

    Args:
        df_raw: Output of :func:`~src.data.make_target.build_target` — the
            labelled frame before feature engineering.
        X_test: Test-split feature matrix returned by
            :func:`~src.data.split.patient_grouped_split`.

    Returns:
        DataFrame with the same rows as the test split but original (text)
        column values, reset to a 0-based integer index.
    """
    return df_raw.reset_index(drop=True).iloc[X_test.index].reset_index(drop=True)


def _age_group(raw_test: pd.DataFrame) -> pd.Series:
    """Map the decade age bins to three coarse clinical brackets."""
    return raw_test["age"].map(_AGE_BUCKET).rename("age_group")


def _gender(raw_test: pd.DataFrame) -> pd.Series:
    """Return gender column, excluding Unknown/Invalid entries."""
    gender = raw_test["gender"].copy()
    invalid = gender.isin(["Unknown/Invalid"])
    if invalid.any():
        logger.info("Excluding %d Unknown/Invalid gender rows from audit.", invalid.sum())
    return gender[~invalid].rename("gender")


def _race(raw_test: pd.DataFrame) -> pd.Series:
    """Return the race column as-is (NaN already filled upstream)."""
    return raw_test["race"].rename("race")


def _admission_type(raw_test: pd.DataFrame) -> pd.Series:
    """Map admission_type_id integers to readable labels."""
    return (
        raw_test["admission_type_id"]
        .map(_ADMISSION_TYPE)
        .fillna("Other")
        .rename("admission_type")
    )


def main() -> None:
    """Run the four-group fairness audit and save results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not _CALIBRATED_PATH.exists():
        logger.error(
            "Calibrated model not found at %s. "
            "Run `python -m src.models.calibrate_lgbm` first.",
            _CALIBRATED_PATH,
        )
        sys.exit(1)

    _FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data pipeline — produce both encoded X_test and raw text rows
    # ------------------------------------------------------------------
    logger.info("Building feature matrix…")
    df_raw = build_target(load_raw())
    X, y, groups = build_features(df_raw)

    X_train, X_val, X_test, y_train, y_val, y_test = patient_grouped_split(X, y, groups)

    # Raw rows for the test split — same positional alignment as X_test.
    raw_test = get_raw_test_rows(df_raw, X_test)

    # ------------------------------------------------------------------
    # Calibrated model predictions
    # ------------------------------------------------------------------
    logger.info("Loading calibrated model from %s", _CALIBRATED_PATH)
    model = joblib.load(_CALIBRATED_PATH)

    y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]
    y_true: np.ndarray = y_test.values
    logger.info(
        "Test set: %d rows  base rate=%.4f  mean pred=%.4f",
        len(y_true), y_true.mean(), y_proba.mean(),
    )

    # ------------------------------------------------------------------
    # Subgroup audits
    # ------------------------------------------------------------------
    audits: list[tuple[str, pd.Series, str]] = [
        ("age",            _age_group(raw_test),   "fairness_age.csv"),
        ("gender",         _gender(raw_test),       "fairness_gender.csv"),
        ("race",           _race(raw_test),         "fairness_race.csv"),
        ("admission type", _admission_type(raw_test), "fairness_admission_type.csv"),
    ]

    results: dict[str, pd.DataFrame] = {}
    csv_paths: list[Path] = []

    for group_name, subgroup, filename in audits:
        logger.info("Auditing subgroup: %s", group_name)

        # Align y_true/y_proba to the subgroup Series index (gender may be filtered).
        idx = subgroup.index
        df_sub = subgroup_metrics(
            y_true=y_true[idx],
            y_proba=y_proba[idx],
            subgroup=subgroup.reset_index(drop=True),
            with_ci=True,
        )

        results[group_name] = df_sub

        csv_path = _FAIRNESS_DIR / filename
        df_sub.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
        logger.info("Saved %s → %s", filename, csv_path)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    for group_name, df_sub in results.items():
        print(format_fairness_table(df_sub, group_name))

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    mlflow.set_experiment("medreadmit-module1")
    with mlflow.start_run(run_name="fairness_audit"):
        mlflow.log_param("model", "lgbm_calibrated")
        mlflow.log_param("n_test_rows", len(y_true))

        for group_name, df_sub in results.items():
            if not df_sub.empty:
                # Log the AUROC spread (max − min) as a single disparity scalar.
                auroc_spread = float(df_sub["auroc"].max() - df_sub["auroc"].min())
                key = group_name.replace(" ", "_")
                mlflow.log_metric(f"{key}_auroc_spread", auroc_spread)

        for csv_path in csv_paths:
            mlflow.log_artifact(str(csv_path), artifact_path="fairness")

        logger.info("Fairness artifacts logged to MLflow run 'fairness_audit'.")


if __name__ == "__main__":
    main()
