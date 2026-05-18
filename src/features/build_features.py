"""Build the model-ready feature matrix from the labelled UCI Diabetes frame."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.features.icd9_grouping import bucket_diagnosis_columns

logger = logging.getLogger(__name__)

# Dropped unconditionally:
#   encounter_id  — row identifier, no signal
#   weight        — 97 % missing; imputing would invent most of the column
#   payer_code    — 40 % missing and not a clinical readmission predictor
_DROP_COLS = ["encounter_id", "weight", "payer_code"]

# Columns where NaN means the test was not ordered — informative missingness.
_NOT_MEASURED_COLS = ["A1Cresult", "max_glu_serum"]

# Rare-category threshold for medical_specialty (fraction of total rows).
_RARE_SPECIALTY_THRESHOLD = 0.01


def apply_feature_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Pure feature transform: drops uninformative columns, encodes, and casts.

    Performs all column-level transforms in the same order as the original
    training pipeline, but does NOT extract ``y`` or ``groups`` — those are the
    responsibility of :func:`build_features`.  This function is safe to call on
    a single-row inference DataFrame as well as the full training set.

    Steps applied (in order):
    1. Drop ``encounter_id``, ``weight``, ``payer_code`` (no-signal / high-missing).
    2. ICD-9 diagnosis bucketing into 9 clinical groups.
    3. Fill ``A1Cresult`` / ``max_glu_serum`` NaN → ``"not_measured"`` (informative).
    4. Fill ``medical_specialty`` NaN → ``"unknown"``; collapse rare categories
       (< 1 % of rows) to ``"other"``.
    5. Fill ``race`` NaN → ``"unknown"``.
    6. Fill any remaining ``diag_2`` / ``diag_3`` NaN → ``"missing"`` (defensive).
    7. One-hot encode all object columns (``drop_first=False``).
    8. Sanitize column names (strip characters rejected by LightGBM).
    9. Cast to ``float32``.

    Args:
        df: Encounter DataFrame containing raw clinical columns.  May or may not
            include ``patient_nbr`` and ``readmitted_30d`` — those are left
            untouched if present.

    Returns:
        Transformed DataFrame of ``float32`` columns ready for model ingestion.
    """
    df = df.copy()

    # --- 1. Drop uninformative / near-empty columns (defensive: skip if absent) ---
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped %s — %d cols remain", cols_to_drop, df.shape[1])

    # --- 2. Bucket ICD-9 diagnosis codes into 9 clinical groups ---
    df = bucket_diagnosis_columns(df)
    logger.info("ICD-9 bucketing applied to diag_1 / diag_2 / diag_3")

    # --- 3. Informative missingness: test-not-ordered sentinel ---
    for col in _NOT_MEASURED_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("not_measured")
    logger.info("Filled NaN in %s → 'not_measured'", _NOT_MEASURED_COLS)

    # --- 4. medical_specialty: impute + collapse rare categories ---
    if "medical_specialty" in df.columns:
        df["medical_specialty"] = df["medical_specialty"].fillna("unknown")
        threshold = _RARE_SPECIALTY_THRESHOLD * len(df)
        counts = df["medical_specialty"].value_counts()
        rare = counts[counts < threshold].index
        df["medical_specialty"] = df["medical_specialty"].replace(rare, "other")
        logger.info(
            "medical_specialty: %d rare specialties (< %.0f%% of rows) collapsed to 'other'",
            len(rare),
            _RARE_SPECIALTY_THRESHOLD * 100,
        )

    # --- 5. race: impute unknown ---
    if "race" in df.columns:
        df["race"] = df["race"].fillna("unknown")

    # --- 6. Defensive fill for diag_2/diag_3 (post-bucketing should be clean) ---
    for col in ("diag_2", "diag_3"):
        if col in df.columns:
            n_remaining = df[col].isna().sum()
            if n_remaining:
                logger.warning(
                    "%s still has %d NaN after bucketing — filling 'missing'", col, n_remaining
                )
                df[col] = df[col].fillna("missing")

    # --- 7. One-hot encode all remaining object columns ---
    obj_cols = df.select_dtypes("object").columns.tolist()
    logger.info("One-hot encoding %d object columns: %s", len(obj_cols), obj_cols)
    df = pd.get_dummies(df, columns=obj_cols, drop_first=False)

    # --- 8. Sanitize names: LightGBM rejects JSON-special characters ---
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_") for col in df.columns]
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature names after sanitization: {dupes}")
    logger.info("Sanitized feature names — %d columns, all unique", df.shape[1])

    # --- 9. Cast to float32 to halve memory vs float64 ---
    df = df.astype(np.float32)
    logger.info("Transform complete: %d rows × %d cols (float32)", *df.shape)
    return df


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Transform the labelled encounter frame into model-ready arrays.

    Extracts ``y`` and ``groups``, then delegates all column-level transforms
    to :func:`apply_feature_transforms`.  The returned DataFrame preserves
    column names so downstream SHAP values map back to interpretable labels.

    Args:
        df: Output of :func:`src.data.make_target.build_target` — encounters
            with ineligible dispositions removed and ``readmitted_30d`` added.

    Returns:
        X: Feature matrix as a DataFrame of float32 columns.
        y: Binary target Series (``readmitted_30d``).
        groups: ``patient_nbr`` values as an ndarray, for use with
            ``GroupShuffleSplit`` to prevent patient-level leakage.
    """
    df = df.reset_index(drop=True).copy()

    # Peel off target and group key before any feature transforms.
    groups: np.ndarray = df["patient_nbr"].values
    y: pd.Series = df["readmitted_30d"]
    df = df.drop(columns=["patient_nbr", "readmitted_30d"])
    logger.info("Extracted target (y) and groups — feature frame: %d cols", df.shape[1])

    df = apply_feature_transforms(df)

    logger.info("Final feature matrix: %d rows × %d cols (float32)", *df.shape)
    return df, y, groups


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.load import load_raw
    from src.data.make_target import build_target

    raw = build_target(load_raw())
    X, y, groups = build_features(raw)  # internally calls apply_feature_transforms

    print(f"\nX shape:      {X.shape}")
    print(f"y shape:      {y.shape}  |  positive rate: {y.mean():.4f}")
    print(f"groups shape: {groups.shape}  |  unique patients: {len(set(groups))}")
    print(f"\nMemory usage: {X.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\nFirst 5 columns: {X.columns[:5].tolist()}")
    print(f"Last  5 columns: {X.columns[-5:].tolist()}")
