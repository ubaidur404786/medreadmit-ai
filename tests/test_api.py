"""Integration tests for src/api/main.py.

All tests run against a real in-process TestClient with the actual
lgbm_calibrated.joblib loaded — no mocks.  The lifespan startup hook is
exercised on the first fixture access and torn down after the module finishes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.feature_alignment import align_to_training_schema
from src.api.main import MODEL_VERSION, app
from src.api.schemas import risk_band_for
from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import patient_grouped_split
from src.features.build_features import build_features

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_MODEL_PATH = _MODELS_DIR / "lgbm_calibrated.joblib"
_MANIFEST_PATH = _MODELS_DIR / "feature_manifest.json"

# Columns present in build_target() output that are NOT part of the API schema.
# These are dropped when constructing a JSON-serialisable request body.
_NON_SCHEMA_COLS = {"patient_nbr", "readmitted_30d", "encounter_id", "weight", "payer_code"}

# Minimal valid request body — used by error-path tests that don't need real data.
_MINIMAL = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "num_lab_procedures": 40,
    "num_medications": 12,
    "number_diagnoses": 5,
}


def _to_json_safe(v: Any) -> Any:
    """Convert a pandas/numpy scalar to a JSON-serialisable Python type.

    - float NaN → None  (maps to JSON null; Pydantic accepts null for Optional fields)
    - numpy scalar → Python native via .item()
    - everything else → unchanged
    """
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_client():
    """Yield a live TestClient with the real model loaded via the lifespan hook.

    Fails (not skips) if the model artifact or manifest is missing so the
    developer sees exactly which file is absent rather than a silent skip.
    """
    if not _MODEL_PATH.exists():
        pytest.fail(
            f"Model artifact not found: {_MODEL_PATH}\n"
            "Run `python -m src.models.calibrate_lgbm` to generate it."
        )
    if not _MANIFEST_PATH.exists():
        pytest.fail(
            f"Feature manifest not found: {_MANIFEST_PATH}\n"
            "Run `python scripts/export_feature_manifest.py` to generate it."
        )
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture(scope="module")
def sample_record() -> dict[str, Any]:
    """Build a single-row request dict from the real test split (row 0).

    Runs the full pipeline once:
      load_raw → build_target → build_features → patient_grouped_split

    Takes row 0 of X_test's index in the PRE-feature-engineering DataFrame,
    drops target/group/schema-excluded columns, and converts pandas/numpy
    types to JSON-safe Python primitives (float NaN → None, numpy int → int).
    """
    labelled = build_target(load_raw()).reset_index(drop=True)
    X, y, groups = build_features(labelled)
    _, _, X_test, _, _, _ = patient_grouped_split(X, y, groups)

    # Row 0 of the held-out test split by position.
    test_idx = X_test.index[0]
    raw_row = labelled.loc[test_idx].to_dict()

    return {k: _to_json_safe(v) for k, v in raw_row.items() if k not in _NON_SCHEMA_COLS}


@pytest.fixture(scope="module")
def expected_proba(app_client: TestClient, sample_record: dict[str, Any]) -> float:
    """Ground-truth probability via batch-mode align + predict_proba.

    Uses app.state populated by the lifespan hook (same model object the API
    serves), so this is the single source of truth for the parity assertion.
    """
    aligned = align_to_training_schema(sample_record, app.state.manifest)
    return float(app.state.model.predict_proba(aligned)[0, 1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_ok(app_client: TestClient) -> None:
    """GET /health should return 200 with all fields present and consistent."""
    resp = app_client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == app.state.manifest["n_features"]
    assert body["model_version"] == MODEL_VERSION


def test_predict_matches_batch(
    app_client: TestClient,
    sample_record: dict[str, Any],
    expected_proba: float,
) -> None:
    """POST /predict must return a probability within 1e-6 of direct batch inference.

    This is THE alignment parity test.  A failure means the HTTP path (Pydantic
    validation → model_dump → align_to_training_schema → predict_proba) produces
    a different feature matrix than the batch path, indicating broken alignment.
    """
    resp = app_client.post("/predict", json=sample_record)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    api_proba = resp.json()["probability"]
    assert abs(api_proba - expected_proba) < 1e-6, (
        f"API probability {api_proba:.8f} differs from batch {expected_proba:.8f} "
        f"by {abs(api_proba - expected_proba):.2e} — alignment is broken"
    )


def test_predict_response_shape(
    app_client: TestClient,
    sample_record: dict[str, Any],
) -> None:
    """POST /predict response must satisfy all field-level contracts."""
    resp = app_client.post("/predict", json=sample_record)
    assert resp.status_code == 200

    body = resp.json()

    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high"}
    assert body["model_version"] == MODEL_VERSION
    assert re.fullmatch(
        r"[0-9a-f]{32}", body["request_id"]
    ), f"request_id is not 32-char hex: {body['request_id']!r}"
    assert body["latency_ms"] > 0.0


def test_predict_missing_required(app_client: TestClient) -> None:
    """POST without 'race' (a required field) must return 422 mentioning 'race'."""
    body_without_race = {k: v for k, v in _MINIMAL.items() if k != "race"}
    resp = app_client.post("/predict", json=body_without_race)
    assert resp.status_code == 422

    # Pydantic detail is a list of error objects; at least one must name 'race'.
    detail = resp.json().get("detail", [])
    assert any(
        "race" in str(err) for err in detail
    ), f"Expected 'race' in error detail, got: {detail}"


def test_predict_bad_type(app_client: TestClient) -> None:
    """POST with time_in_hospital='five' (str instead of int) must return 422."""
    bad_body = {**_MINIMAL, "time_in_hospital": "five"}
    resp = app_client.post("/predict", json=bad_body)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "prob,expected_band",
    [
        (0.05, "low"),
        (0.09, "low"),
        (0.10, "moderate"),
        (0.15, "moderate"),
        (0.29, "moderate"),
        (0.30, "high"),
        (0.50, "high"),
        (0.99, "high"),
    ],
)
def test_risk_band_thresholds(prob: float, expected_band: str) -> None:
    """risk_band_for must bucket probabilities at the documented thresholds.

    Thresholds: < 0.10 → low, 0.10–0.30 → moderate, >= 0.30 → high.
    Anchored to the ~11 % positive-rate base rate (WEEK1_RESULTS.md).
    """
    assert (
        risk_band_for(prob) == expected_band
    ), f"risk_band_for({prob}) returned {risk_band_for(prob)!r}, expected {expected_band!r}"


def test_request_ids_unique(
    app_client: TestClient,
    sample_record: dict[str, Any],
) -> None:
    """Five sequential /predict calls must produce five distinct request_ids."""
    ids = [app_client.post("/predict", json=sample_record).json()["request_id"] for _ in range(5)]
    assert len(set(ids)) == 5, f"Duplicate request_ids detected: {ids}"


def test_predict_returns_shap(
    app_client: TestClient,
    sample_record: dict[str, Any],
) -> None:
    """POST /predict must return exactly 5 SHAP contributions, correctly structured.

    Validates:
    - top_features has length 5
    - each element carries feature, shap_value, feature_value
    - every feature name is in manifest["feature_columns"]
    - contributions are sorted by descending |shap_value|
    """
    resp = app_client.post("/predict", json=sample_record)
    assert resp.status_code == 200

    body = resp.json()
    top = body["top_features"]

    assert len(top) == 5, f"Expected 5 top_features, got {len(top)}"

    feature_columns = set(app.state.manifest["feature_columns"])
    for i, contrib in enumerate(top):
        assert "feature" in contrib, f"Entry {i} missing 'feature'"
        assert "shap_value" in contrib, f"Entry {i} missing 'shap_value'"
        assert "feature_value" in contrib, f"Entry {i} missing 'feature_value'"
        assert isinstance(contrib["shap_value"], float), (
            f"Entry {i} shap_value is {type(contrib['shap_value'])}, expected float"
        )
        assert isinstance(contrib["feature_value"], float), (
            f"Entry {i} feature_value is {type(contrib['feature_value'])}, expected float"
        )
        assert contrib["feature"] in feature_columns, (
            f"Entry {i} feature {contrib['feature']!r} not in manifest feature_columns"
        )

    abs_shap = [abs(c["shap_value"]) for c in top]
    assert abs_shap == sorted(abs_shap, reverse=True), (
        f"top_features not sorted by descending |shap_value|: {abs_shap}"
    )


def test_predict_top_feature_is_known(app_client: TestClient) -> None:
    """A high-risk record should rank number_inpatient in its top-5 SHAP features.

    number_inpatient was the #1 feature by mean |SHAP| on the held-out test set
    (Week 1 findings, WEEK1_RESULTS.md).  Sending a record with a very high
    number_inpatient (5 prior inpatient visits) and low competing signals
    should surface it in the top-5 every time.
    """
    high_risk_record = {
        **_MINIMAL,
        "number_inpatient": 5,   # strong positive predictor (#1 by mean |SHAP|)
        "number_emergency": 0,
        "number_outpatient": 0,
        "num_procedures": 0,
        "diag_1": "250.83",
        "diag_2": "428",
        "diag_3": "V58",
    }
    resp = app_client.post("/predict", json=high_risk_record)
    assert resp.status_code == 200

    body = resp.json()
    assert body["risk_band"] == "high", (
        f"Expected high-risk band for number_inpatient=5, got {body['risk_band']!r}"
    )

    top_feature_names = [c["feature"] for c in body["top_features"]]
    assert "number_inpatient" in top_feature_names, (
        f"number_inpatient not in top-5 features for high-risk record: {top_feature_names}"
    )


# ---------------------------------------------------------------------------
# Batch endpoint tests
# ---------------------------------------------------------------------------

# Three structurally distinct records used across batch tests.
_BATCH_RECORDS = [
    _MINIMAL,
    {**_MINIMAL, "number_inpatient": 2, "num_procedures": 1, "diag_1": "250.83"},
    {**_MINIMAL, "number_inpatient": 5, "number_emergency": 1, "diag_1": "428", "diag_2": "V58"},
]


def test_batch_predict_basic(app_client: TestClient) -> None:
    """POST /predict/batch with 3 records must return 3 predictions at indices [0, 1, 2]."""
    resp = app_client.post("/predict/batch", json={"encounters": _BATCH_RECORDS})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    body = resp.json()
    assert body["n_processed"] == 3
    assert len(body["predictions"]) == 3
    assert [p["index"] for p in body["predictions"]] == [0, 1, 2]

    for pred in body["predictions"]:
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["risk_band"] in {"low", "moderate", "high"}
        assert len(pred["top_features"]) == 5


def test_batch_matches_single(app_client: TestClient) -> None:
    """Probabilities from /predict/batch must match individual /predict calls within 1e-6.

    This is the batch parity test: proves that pd.concat + single predict_proba
    produces identical scores to per-row inference.
    """
    single_probas = [
        app_client.post("/predict", json=rec).json()["probability"] for rec in _BATCH_RECORDS
    ]

    batch_resp = app_client.post("/predict/batch", json={"encounters": _BATCH_RECORDS})
    assert batch_resp.status_code == 200
    batch_probas = [p["probability"] for p in batch_resp.json()["predictions"]]

    for i, (single, batch) in enumerate(zip(single_probas, batch_probas)):
        assert abs(single - batch) < 1e-6, (
            f"Index {i}: single={single:.8f} batch={batch:.8f} "
            f"delta={abs(single - batch):.2e} — batch alignment is broken"
        )


def test_batch_empty_rejected(app_client: TestClient) -> None:
    """POST /predict/batch with an empty encounters list must return 422."""
    resp = app_client.post("/predict/batch", json={"encounters": []})
    assert resp.status_code == 422


def test_batch_too_large_rejected(app_client: TestClient) -> None:
    """POST /predict/batch with 101 encounters must return 422 (max_length=100)."""
    resp = app_client.post("/predict/batch", json={"encounters": [_MINIMAL] * 101})
    assert resp.status_code == 422


def test_batch_latency_reasonable(app_client: TestClient, sample_record: dict[str, Any]) -> None:
    """Batch inference must be faster per record than a single /predict call.

    Sends 10 records in one batch and compares mean_latency_per_record_ms against
    the latency_ms of a single /predict call.  Batching model scoring and SHAP
    explanation over N rows must have lower per-record overhead than N sequential
    single calls.
    """
    single_resp = app_client.post("/predict", json=sample_record)
    assert single_resp.status_code == 200
    single_latency_ms = single_resp.json()["latency_ms"]

    batch_resp = app_client.post(
        "/predict/batch", json={"encounters": [sample_record] * 10}
    )
    assert batch_resp.status_code == 200
    mean_ms = batch_resp.json()["mean_latency_per_record_ms"]

    assert mean_ms < single_latency_ms, (
        f"Batch mean latency ({mean_ms:.2f} ms/record) is not faster than "
        f"single /predict latency ({single_latency_ms:.2f} ms) — "
        "batching is not providing the expected speedup"
    )
