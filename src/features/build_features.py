"""Build the model-ready feature matrix from the labelled UCI Diabetes frame."""

from __future__ import annotations

import logging

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


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Transform the labelled encounter frame into model-ready arrays.

    Performs column pruning, ICD-9 bucketing, categorical imputation,
    rare-category collapsing, one-hot encoding, and float32 casting.
    The returned DataFrame preserves column names so downstream SHAP
    values map back to interpretable feature labels.

    Args:
        df: Output of :func:`src.data.make_target.build_target` — encounters
            with ineligible dispositions removed and ``readmitted_30d`` added.

    Returns:
        X: Feature matrix as a DataFrame of float32 columns.
        y: Binary target Series (``readmitted_30d``).
        groups: ``patient_nbr`` values as an ndarray, for use with
            ``GroupShuffleSplit`` to prevent patient-level leakage.
    """
    df = df.reset_index(drop=True)
    df = df.copy()
    
    # --- 1. Drop uninformative / near-empty columns ---
    df = df.drop(columns=_DROP_COLS)
    logger.info("Dropped %s — %d cols remain", _DROP_COLS, df.shape[1])

    # --- 2. Peel off target and group key before any feature transforms ---
    groups: np.ndarray = df["patient_nbr"].values
    y: pd.Series = df["readmitted_30d"]
    df = df.drop(columns=["patient_nbr", "readmitted_30d"])
    logger.info("Extracted target (y) and groups — feature frame: %d cols", df.shape[1])

    # --- 3. Bucket ICD-9 diagnosis codes into 9 clinical groups ---
    df = bucket_diagnosis_columns(df)
    logger.info("ICD-9 bucketing applied to diag_1 / diag_2 / diag_3")

    # --- 4. Informative missingness: test-not-ordered sentinel ---
    for col in _NOT_MEASURED_COLS:
        df[col] = df[col].fillna("not_measured")
    logger.info("Filled NaN in %s → 'not_measured'", _NOT_MEASURED_COLS)

    # --- 5. medical_specialty: impute + collapse rare categories ---
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

    # --- 6. race: impute unknown ---
    df["race"] = df["race"].fillna("unknown")

    # --- 7. Defensive fill for diag_2/diag_3 (post-bucketing should be clean) ---
    for col in ("diag_2", "diag_3"):
        n_remaining = df[col].isna().sum()
        if n_remaining:
            logger.warning("%s still has %d NaN after bucketing — filling 'missing'", col, n_remaining)
            df[col] = df[col].fillna("missing")

    # --- 8. One-hot encode all remaining object columns ---
    obj_cols = df.select_dtypes("object").columns.tolist()
    logger.info("One-hot encoding %d object columns: %s", len(obj_cols), obj_cols)
    df = pd.get_dummies(df, columns=obj_cols, drop_first=False)

    # --- 9. Cast to float32 to halve memory vs float64 ---
    df = df.astype(np.float32)

    logger.info("Final feature matrix: %d rows × %d cols (float32)", *df.shape)
    return df, y, groups


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.load import load_raw
    from src.data.make_target import build_target

    raw = build_target(load_raw())
    X, y, groups = build_features(raw)

    print(f"\nX shape:      {X.shape}")
    print(f"y shape:      {y.shape}  |  positive rate: {y.mean():.4f}")
    print(f"groups shape: {groups.shape}  |  unique patients: {len(set(groups))}")
    print(f"\nMemory usage: {X.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\nFirst 5 columns: {X.columns[:5].tolist()}")
    print(f"Last  5 columns: {X.columns[-5:].tolist()}")
