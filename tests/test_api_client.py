"""Unit tests for frontend/api_client.py using httpx.MockTransport.

No real server is needed — every test supplies a handler function to
MockTransport that returns a pre-built httpx.Response.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from frontend.api_client import MedReadmitAPI, MedReadmitAPIError

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

_MINIMAL_ENCOUNTER: dict[str, Any] = {
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

_HEALTH_BODY: dict[str, Any] = {
    "status": "ok",
    "model_loaded": True,
    "n_features": 154,
    "model_version": "lgbm_calibrated_v1",
}

_PREDICT_BODY: dict[str, Any] = {
    "probability": 0.13,
    "risk_band": "moderate",
    "model_version": "lgbm_calibrated_v1",
    "request_id": "a" * 32,
    "latency_ms": 62.4,
    "top_features": [
        {"feature": "number_inpatient", "shap_value": 0.41, "feature_value": 2.0},
        {"feature": "discharge_disposition_id", "shap_value": -0.21, "feature_value": 1.0},
        {"feature": "time_in_hospital", "shap_value": 0.18, "feature_value": 3.0},
        {"feature": "num_medications", "shap_value": 0.12, "feature_value": 12.0},
        {"feature": "admission_type_id", "shap_value": -0.09, "feature_value": 1.0},
    ],
}

_BATCH_BODY: dict[str, Any] = {
    "predictions": [
        {
            "index": 0,
            "probability": 0.07,
            "risk_band": "low",
            "top_features": [],
        },
        {
            "index": 1,
            "probability": 0.22,
            "risk_band": "moderate",
            "top_features": [],
        },
    ],
    "n_processed": 2,
    "model_version": "lgbm_calibrated_v1",
    "request_id": "b" * 32,
    "latency_ms": 65.1,
    "mean_latency_per_record_ms": 5.8,
}


def _make_client(handler: Any) -> MedReadmitAPI:
    """Return a MedReadmitAPI wired to a MockTransport backed by *handler*."""
    return MedReadmitAPI(
        base_url="http://testserver",
        _transport=httpx.MockTransport(handler),
    )


def _json_response(body: dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_returns_expected_shape() -> None:
    """health() must return all four documented keys with correct types."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return _json_response(_HEALTH_BODY)

    api = _make_client(handler)
    result = api.health()

    assert result["status"] == "ok"
    assert result["model_loaded"] is True
    assert isinstance(result["n_features"], int)
    assert isinstance(result["model_version"], str)


def test_predict_raises_on_422_with_readable_message() -> None:
    """predict() must raise MedReadmitAPIError with status 422 and a message that
    names the offending field when the server returns a Pydantic validation error."""

    pydantic_detail = [
        {
            "type": "missing",
            "loc": ["body", "race"],
            "msg": "Field required",
            "url": "https://errors.pydantic.dev/2.0/v/missing",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"detail": pydantic_detail}, status_code=422)

    api = _make_client(handler)
    with pytest.raises(MedReadmitAPIError) as exc_info:
        api.predict({k: v for k, v in _MINIMAL_ENCOUNTER.items() if k != "race"})

    err = exc_info.value
    assert err.status_code == 422
    assert "race" in err.detail, (
        f"Expected 'race' in error detail, got: {err.detail!r}"
    )


def test_predict_raises_on_500() -> None:
    """predict() must raise MedReadmitAPIError with status 500 on server errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            {"detail": "Internal server error", "request_id": "c" * 32},
            status_code=500,
        )

    api = _make_client(handler)
    with pytest.raises(MedReadmitAPIError) as exc_info:
        api.predict(_MINIMAL_ENCOUNTER)

    err = exc_info.value
    assert err.status_code == 500
    assert "Internal server error" in err.detail


def test_predict_batch_accepts_list_returns_list() -> None:
    """predict_batch() must POST encounters wrapped in {'encounters': [...]},
    then unwrap the server response and return predictions as a list."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/predict/batch"
        body = json.loads(request.content)
        # Client must wrap the list under the 'encounters' key.
        assert "encounters" in body, f"Missing 'encounters' key in body: {list(body)}"
        assert len(body["encounters"]) == 2
        return _json_response(_BATCH_BODY)

    api = _make_client(handler)
    result = api.predict_batch([_MINIMAL_ENCOUNTER, _MINIMAL_ENCOUNTER])

    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 2
    assert result["n_processed"] == 2
    # Each prediction must carry index, probability, risk_band.
    for pred in result["predictions"]:
        assert "index" in pred
        assert "probability" in pred
        assert pred["risk_band"] in ("low", "moderate", "high")
