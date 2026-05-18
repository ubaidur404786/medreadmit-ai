"""Tests for src/api/feature_alignment.py — single-record inference alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.api.feature_alignment import align_to_training_schema
from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import patient_grouped_split
from src.features.build_features import build_features

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "models" / "feature_manifest.json"

# ---------------------------------------------------------------------------
# Fixtures — loaded once per module (expensive: full 99k-row pipeline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline_data(manifest) -> tuple:
    """Runs the full pipeline once and returns (labelled_df, X_test).

    ``labelled_df`` is reset-indexed to match the internal indexing that
    ``build_features`` uses, so ``labelled_df.loc[idx]`` correctly retrieves
    the pre-engineering row for any ``idx`` in ``X_test.index``.
    """
    labelled = build_target(load_raw())
    labelled = labelled.reset_index(drop=True)
    X, y, groups = build_features(labelled)
    _, _, X_test, _, _, _ = patient_grouped_split(X, y, groups)
    return labelled, X_test


# ---------------------------------------------------------------------------
# Shared minimal record for unit tests that don't need real data
# ---------------------------------------------------------------------------

_MINIMAL_RECORD: dict = {
    # Required demographic / administrative fields
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 12,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 5,
    # Diagnosis codes — drive ICD-9 bucketing
    "diag_1": "250.83",
    "diag_2": "428",
    "diag_3": "V58",
    # Lab results (None → "not_measured" sentinel in pipeline)
    "max_glu_serum": None,
    "A1Cresult": None,
    # Specialty (common — in training vocabulary)
    "medical_specialty": "InternalMedicine",
    # Medication columns
    "metformin": "Steady",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "Steady",
    # Hyphenated combination drugs (match raw column names)
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "Yes",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_round_trip_matches_test_set(manifest, pipeline_data) -> None:
    """Raw clinical record → align → should equal the training pipeline's X_test row.

    The round-trip is exact for records whose medical_specialty is either null
    (→ "unknown") or a common specialty (one that appeared in ≥ 1 % of training
    encounters and therefore has its own one-hot column in the manifest).

    Rare specialties (< 1 % → collapsed to "other" at training time) cannot
    round-trip exactly because the single-row rare-category collapse threshold
    is 0.01 * 1 = 0.01, meaning nothing is ever collapsed; this is a known
    limitation documented in feature_alignment.py.  Rows with rare specialties
    are excluded from this test.
    """
    labelled, X_test = pipeline_data

    # Derive the set of specialty category names that have their own manifest column.
    known_specialty_cats = {
        c[len("medical_specialty_") :]
        for c in manifest["feature_columns"]
        if c.startswith("medical_specialty_")
    }

    # Filter X_test to rows whose raw specialty is null or in the known set.
    raw_specialties = labelled.loc[X_test.index, "medical_specialty"]
    eligible_mask = raw_specialties.isna() | raw_specialties.isin(known_specialty_cats)
    eligible_indices = X_test.index[eligible_mask]

    # Sample 5 rows deterministically from the eligible set.
    sample_indices = eligible_indices.to_series().sample(5, random_state=42).index

    exclude = {"patient_nbr", "readmitted_30d"}

    for idx in sample_indices:
        raw_record = {k: v for k, v in labelled.loc[idx].to_dict().items() if k not in exclude}
        result = align_to_training_schema(raw_record, manifest)
        expected = X_test.loc[[idx]]

        assert (
            result.shape == expected.shape
        ), f"Row {idx}: shape mismatch {result.shape} vs {expected.shape}"
        assert list(result.columns) == list(expected.columns), f"Row {idx}: column order mismatch"
        np.testing.assert_allclose(
            result.values,
            expected.values,
            atol=1e-6,
            err_msg=f"Row {idx}: values do not match training pipeline output",
        )


def test_unseen_category_fills_zero(manifest) -> None:
    """A truly unseen medical_specialty should produce all-zero specialty columns.

    "NonExistentSpecialty" was never in the training data, so apply_feature_transforms
    creates a phantom one-hot column (medical_specialty_NonExistentSpecialty) that
    reindex silently drops, leaving all medical_specialty_* columns at 0.
    This is the correct behaviour: the model gets no specialty signal rather than
    a misleading activation of the "other" (known-but-rare) category.
    """
    record = {**_MINIMAL_RECORD, "medical_specialty": "NonExistentSpecialty"}
    result = align_to_training_schema(record, manifest)

    spec_cols = [c for c in manifest["feature_columns"] if c.startswith("medical_specialty_")]
    assert spec_cols, "Manifest has no medical_specialty_* columns — manifest may be stale"

    specialty_values = result[spec_cols].values.flatten()
    assert np.all(specialty_values == 0.0), (
        f"Expected all medical_specialty_* = 0 for unseen specialty, "
        f"got non-zero in: {[c for c in spec_cols if result[c].iloc[0] != 0.0]}"
    )


def test_missing_a1c_marks_not_measured(manifest) -> None:
    """A1Cresult=None (not ordered) should activate the not_measured one-hot column."""
    record = {**_MINIMAL_RECORD, "A1Cresult": None}
    result = align_to_training_schema(record, manifest)

    # The sanitized column name: "A1Cresult_not_measured"
    col = "A1Cresult_not_measured"
    assert col in result.columns, f"Expected column '{col}' in output — manifest may be out of date"
    assert result[col].iloc[0] == pytest.approx(
        1.0
    ), f"Expected {col} = 1.0 when A1Cresult is None, got {result[col].iloc[0]}"

    # All other A1Cresult_* columns must be 0.
    other_a1c = [c for c in manifest["feature_columns"] if c.startswith("A1Cresult_") and c != col]
    for c in other_a1c:
        assert result[c].iloc[0] == pytest.approx(
            0.0
        ), f"Expected {c} = 0.0 when A1Cresult is None, got {result[c].iloc[0]}"


def test_output_shape(manifest) -> None:
    """Any valid record must produce a DataFrame of shape (1, n_features)."""
    result = align_to_training_schema(_MINIMAL_RECORD, manifest)

    n_features = manifest["n_features"]
    assert result.shape == (1, n_features), f"Expected (1, {n_features}), got {result.shape}"
    assert result.dtypes.unique().tolist() == [
        np.float32
    ], f"Expected all float32, got {result.dtypes.unique().tolist()}"
    assert (
        list(result.columns) == manifest["feature_columns"]
    ), "Output column order does not match manifest"
