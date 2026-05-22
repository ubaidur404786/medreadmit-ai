"""Unit tests for src/api/explainer.py.

Exercises APIExplainer directly against the real model artifact — no FastAPI
TestClient, no HTTP round-trip.  The module-scoped fixtures load heavy objects
once and share them across all tests in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.api.explainer import APIExplainer
from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import patient_grouped_split
from src.features.build_features import build_features

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_MODEL_PATH = _MODELS_DIR / "lgbm_calibrated.joblib"
_MANIFEST_PATH = _MODELS_DIR / "feature_manifest.json"

# Positional row indices to pull from the held-out test split.
_TEST_ROW_POSITIONS = [0, 100, 1000]


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    if not _MODEL_PATH.exists():
        pytest.fail(
            f"Model artifact not found: {_MODEL_PATH}\n"
            "Run `python -m src.models.calibrate_lgbm` to generate it."
        )
    return joblib.load(_MODEL_PATH)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        pytest.fail(
            f"Feature manifest not found: {_MANIFEST_PATH}\n"
            "Run `python scripts/export_feature_manifest.py` to generate it."
        )
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def explainer(model, manifest) -> APIExplainer:
    return APIExplainer(model, manifest["feature_columns"])


@pytest.fixture(scope="module")
def sample_X(manifest):
    """Three rows from the held-out test split, aligned to the manifest schema.

    Runs the full pipeline once (load_raw → build_target → build_features →
    patient_grouped_split) and slices positional rows [0, 100, 1000] from X_test.
    X_test already has apply_feature_transforms applied, so columns are
    consistent with training.  Reindex to the manifest column order so the
    DataFrame is a valid input for APIExplainer.explain().
    """
    labelled = build_target(load_raw()).reset_index(drop=True)
    X, y, groups = build_features(labelled)
    _, _, X_test, _, _, _ = patient_grouped_split(X, y, groups)

    rows = X_test.iloc[_TEST_ROW_POSITIONS]
    # Align to manifest column order (fills zeros for any absent columns).
    return rows.reindex(columns=manifest["feature_columns"], fill_value=np.float32(0)).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_explainer_construction_succeeds(explainer: APIExplainer) -> None:
    """APIExplainer must build without error from the real model artifact."""
    assert explainer is not None
    assert hasattr(explainer, "_explainer")
    assert hasattr(explainer, "_feature_names")


def test_explain_returns_correct_shape(explainer: APIExplainer, sample_X) -> None:
    """explain(sample_X, top_k=5) must return a list of 3 lists, each of length 5."""
    result = explainer.explain(sample_X, top_k=5)

    assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
    assert len(result) == len(_TEST_ROW_POSITIONS), (
        f"Expected {len(_TEST_ROW_POSITIONS)} rows, got {len(result)}"
    )
    for i, row_contribs in enumerate(result):
        assert isinstance(row_contribs, list), f"Row {i}: expected list, got {type(row_contribs)}"
        assert len(row_contribs) == 5, f"Row {i}: expected 5 contributions, got {len(row_contribs)}"


def test_top_k_respected(explainer: APIExplainer, sample_X, manifest: dict) -> None:
    """top_k parameter must control the number of returned contributions per row."""
    n_features = manifest["n_features"]

    result_3 = explainer.explain(sample_X, top_k=3)
    for i, row in enumerate(result_3):
        assert len(row) == 3, f"top_k=3: row {i} has {len(row)} contributions"

    # top_k larger than n_features is clamped to n_features naturally by array slicing.
    oversized_k = 20
    expected_k = min(oversized_k, n_features)
    result_20 = explainer.explain(sample_X, top_k=oversized_k)
    for i, row in enumerate(result_20):
        assert len(row) == expected_k, (
            f"top_k={oversized_k}: row {i} has {len(row)} contributions, expected {expected_k}"
        )


def test_shap_values_sorted(explainer: APIExplainer, sample_X) -> None:
    """Contributions for each row must be sorted by descending |shap_value|."""
    result = explainer.explain(sample_X, top_k=5)

    for i, row_contribs in enumerate(result):
        abs_shap = [abs(c["shap_value"]) for c in row_contribs]
        assert abs_shap == sorted(abs_shap, reverse=True), (
            f"Row {i} contributions are not sorted by descending |shap_value|: {abs_shap}"
        )


def test_features_are_in_manifest(
    explainer: APIExplainer, sample_X, manifest: dict
) -> None:
    """Every feature name in every contribution must appear in manifest['feature_columns']."""
    feature_columns = set(manifest["feature_columns"])
    result = explainer.explain(sample_X, top_k=5)

    for i, row_contribs in enumerate(result):
        for contrib in row_contribs:
            assert contrib["feature"] in feature_columns, (
                f"Row {i}: feature {contrib['feature']!r} not in manifest feature_columns"
            )


def test_feature_values_match_input(explainer: APIExplainer, sample_X) -> None:
    """feature_value in each contribution must equal the corresponding cell in sample_X.

    Compares row 0 only.  Both sides are float32-derived, so tolerance is set
    to account for float32→float64 round-trip (relative error < 1e-5).
    """
    result = explainer.explain(sample_X, top_k=5)
    row_contribs = result[0]  # contributions for sample_X row 0

    for contrib in row_contribs:
        feature = contrib["feature"]
        returned_value = contrib["feature_value"]
        actual_value = float(sample_X.iloc[0][feature])

        assert abs(returned_value - actual_value) <= 1e-5 * max(abs(actual_value), 1e-9), (
            f"feature_value mismatch for '{feature}': "
            f"returned {returned_value}, expected {actual_value}"
        )
