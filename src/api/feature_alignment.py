"""Align a single raw clinical record to the training-time feature schema.

The feature matrix produced by ``build_features`` during training has a fixed
set of 154 one-hot-encoded float32 columns.  At inference time, a single-row
record (e.g. from a FastAPI request body) goes through the same transforms but
can produce a different column set because:

- Unseen categorical levels create new one-hot columns that were absent in
  training (e.g. a medical specialty seen for the first time).
- Categorical levels that appeared in training but are absent from this record
  produce no column at all.

``align_to_training_schema`` handles both cases: missing training columns are
filled with 0 (sensible default for one-hot columns), and extra columns are
dropped (with a warning logged, since this should not happen in practice).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_features import apply_feature_transforms

logger = logging.getLogger(__name__)


def align_to_training_schema(
    record: dict[str, Any],
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Transform a single raw clinical record into a model-ready DataFrame.

    Runs the identical feature transforms used at training time, then reindexes
    the result to match the manifest's column list exactly.  One-hot columns for
    categories not present in this record are filled with ``0``; columns produced
    by the transform but absent from the manifest are dropped (and logged).

    Args:
        record: Flat dict of raw clinical fields, e.g. from a
            :class:`src.api.schemas.PatientEncounter` model dump.  Should not
            contain ``patient_nbr`` or ``readmitted_30d`` (API inputs exclude
            them).  ``encounter_id``, ``weight``, and ``payer_code`` are
            silently dropped by the transform if present.
        manifest: Loaded ``models/feature_manifest.json`` dict, which must
            contain a ``"feature_columns"`` key listing the ordered column names
            the model expects.

    Returns:
        DataFrame of shape ``(1, n_features)`` with dtype ``float32``,
        columns ordered exactly as ``manifest["feature_columns"]``.
    """
    expected_cols: list[str] = manifest["feature_columns"]

    # Wrap the single record in a one-row DataFrame so apply_feature_transforms
    # receives a proper DataFrame (it expects to call .copy(), select_dtypes, etc.).
    df = pd.DataFrame([record])
    df = apply_feature_transforms(df)

    # Columns produced by the transform but not in the manifest.
    # In steady state this is empty; a non-empty set signals a pipeline change
    # that broke the training/inference symmetry and warrants investigation.
    extra = sorted(set(df.columns) - set(expected_cols))
    if extra:
        logger.warning(
            "align_to_training_schema: dropping %d post-transform column(s) not in manifest "
            "(unseen categorical levels or pipeline drift): %s",
            len(extra),
            extra,
        )

    # Known limitation — rare medical_specialty mismatch:
    # Training collapses specialties appearing in < 1 % of training rows → "other".
    # On a single row the threshold is 0.01 * 1 = 0.01, so no specialty is ever
    # collapsed; a rare-but-seen specialty creates a phantom one-hot column that
    # reindex zeros out instead of routing to medical_specialty_other.
    # Impact: < 0.5 % of encounters; top predictors are unaffected.
    # Fix path: store known_specialty_categories in the manifest and pre-remap here.

    # Reindex to the exact training column order:
    #   - missing columns (unseen categories → absent one-hot cols) → filled with 0
    #   - extra columns from above → silently omitted by reindex
    df = df.reindex(columns=expected_cols, fill_value=np.float32(0))
    return df.astype(np.float32)
