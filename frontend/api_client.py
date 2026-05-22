"""Thin synchronous wrapper around the MedReadmit FastAPI prediction endpoints.

Usage
-----
>>> from frontend.api_client import MedReadmitAPI, MedReadmitAPIError
>>> api = MedReadmitAPI()                        # reads MEDREADMIT_API_BASE env var
>>> api.health()
{'status': 'ok', 'model_loaded': True, 'n_features': 154, 'model_version': '...'}
>>> api.predict({"race": "Caucasian", "gender": "Female", ...})
{'probability': 0.13, 'risk_band': 'moderate', ...}
>>> api.predict_batch([encounter_1, encounter_2])
{'predictions': [...], 'n_processed': 2, ...}

Set the API base URL via the environment variable MEDREADMIT_API_BASE or pass it
explicitly to the constructor.  The default assumes uvicorn running on localhost:8000.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class MedReadmitAPIError(Exception):
    """Raised when the MedReadmit API returns a non-2xx response.

    Attributes
    ----------
    status_code:
        HTTP status code returned by the server (e.g. 422, 500).
    detail:
        Human-readable error message extracted from the response body.
        Falls back to the raw response text if the body is not JSON.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MedReadmitAPI:
    """Synchronous HTTP client for the MedReadmit prediction API.

    Parameters
    ----------
    base_url:
        Root URL of the running API, e.g. ``"http://localhost:8000"``.
        Defaults to the ``MEDREADMIT_API_BASE`` environment variable, then
        ``"http://localhost:8000"``.
    timeout:
        Per-request timeout in seconds (default 5.0).  Raise for batch
        requests over slow networks or very large batches.

    Examples
    --------
    >>> api = MedReadmitAPI()
    >>> api.predict({"race": "Caucasian", "gender": "Female", "age": "[60-70)",
    ...              "admission_type_id": 1, "discharge_disposition_id": 1,
    ...              "admission_source_id": 7, "time_in_hospital": 3,
    ...              "num_lab_procedures": 40, "num_medications": 12,
    ...              "number_diagnoses": 5})
    {'probability': 0.09, 'risk_band': 'low', ...}
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 5.0,
        *,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("MEDREADMIT_API_BASE", "http://localhost:8000")
        ).rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=_transport,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a request and return the parsed JSON body.

        Raises :class:`MedReadmitAPIError` on any non-2xx response.
        Logs the round-trip latency at DEBUG level.
        """
        t0 = time.perf_counter()
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise MedReadmitAPIError(
                status_code=0,
                detail=f"Request timed out: {exc}",
            ) from exc
        except httpx.RequestError as exc:
            raise MedReadmitAPIError(
                status_code=0,
                detail=f"Connection error: {exc}",
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "%s %s -> %d  %.1f ms",
            method.upper(),
            path,
            resp.status_code,
            latency_ms,
        )

        if resp.is_success:
            return resp.json()

        # Extract a readable error message from the response body.
        try:
            body = resp.json()
            # FastAPI validation errors are a list under "detail".
            raw_detail = body.get("detail", body)
            if isinstance(raw_detail, list):
                # Pydantic v2 format: [{loc, msg, type}, ...]
                detail = "; ".join(
                    f"{' -> '.join(str(p) for p in e.get('loc', []))}: {e.get('msg', e)}"
                    if isinstance(e, dict)
                    else str(e)
                    for e in raw_detail
                )
            else:
                detail = str(raw_detail)
        except Exception:
            detail = resp.text or f"HTTP {resp.status_code}"

        raise MedReadmitAPIError(status_code=resp.status_code, detail=detail)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return the service health and loaded model metadata.

        Returns
        -------
        dict with keys:
            ``status`` ("ok" or "degraded"), ``model_loaded`` (bool),
            ``n_features`` (int), ``model_version`` (str).

        Raises
        ------
        MedReadmitAPIError
            If the server returns a non-2xx response.

        Examples
        --------
        >>> api = MedReadmitAPI()
        >>> h = api.health()
        >>> assert h["status"] == "ok"
        >>> assert h["model_loaded"] is True
        """
        return self._request("GET", "/health")

    def predict(self, encounter: dict[str, Any]) -> dict[str, Any]:
        """Score a single patient encounter.

        Parameters
        ----------
        encounter:
            Flat dict matching the ``PatientEncounter`` Pydantic schema.
            Required keys: ``race``, ``gender``, ``age``,
            ``admission_type_id``, ``discharge_disposition_id``,
            ``admission_source_id``, ``time_in_hospital``,
            ``num_lab_procedures``, ``num_medications``, ``number_diagnoses``.
            All other keys are optional (omit or set to ``None``).

        Returns
        -------
        dict with keys:
            ``probability`` (float 0–1), ``risk_band`` ("low"/"moderate"/"high"),
            ``model_version`` (str), ``request_id`` (str), ``latency_ms`` (float),
            ``top_features`` (list of dicts with ``feature``, ``shap_value``,
            ``feature_value``).

        Raises
        ------
        MedReadmitAPIError
            422 if the encounter fails schema validation (missing required field,
            wrong type, etc.).  500 on unexpected server errors.

        Examples
        --------
        >>> api = MedReadmitAPI()
        >>> result = api.predict({
        ...     "race": "Caucasian", "gender": "Female", "age": "[70-80)",
        ...     "admission_type_id": 1, "discharge_disposition_id": 6,
        ...     "admission_source_id": 7, "time_in_hospital": 6,
        ...     "num_lab_procedures": 22, "num_medications": 14,
        ...     "number_diagnoses": 8, "number_inpatient": 5,
        ... })
        >>> result["risk_band"] in ("low", "moderate", "high")
        True
        """
        return self._request("POST", "/predict", json=encounter)

    def predict_batch(self, encounters: list[dict[str, Any]]) -> dict[str, Any]:
        """Score a batch of 1–100 patient encounters in a single API call.

        Parameters
        ----------
        encounters:
            List of encounter dicts, each matching the ``PatientEncounter``
            schema (same format as :meth:`predict`).  Must contain 1–100 items.

        Returns
        -------
        dict with keys:
            ``predictions`` (list of per-encounter results, each with ``index``,
            ``probability``, ``risk_band``, ``top_features``),
            ``n_processed`` (int), ``model_version`` (str), ``request_id`` (str),
            ``latency_ms`` (float), ``mean_latency_per_record_ms`` (float).

        Raises
        ------
        MedReadmitAPIError
            422 if any encounter fails validation or the list is empty / exceeds
            100 items.  500 on unexpected server errors.

        Examples
        --------
        >>> api = MedReadmitAPI()
        >>> records = [encounter_a, encounter_b, encounter_c]
        >>> batch = api.predict_batch(records)
        >>> len(batch["predictions"]) == 3
        True
        >>> batch["predictions"][0]["risk_band"] in ("low", "moderate", "high")
        True
        """
        return self._request("POST", "/predict/batch", json={"encounters": encounters})

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MedReadmitAPI":
        return self

    def __exit__(self, *args: object) -> None:
        self._client.close()

    def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        self._client.close()
