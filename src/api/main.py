"""FastAPI application for 30-day hospital readmission risk prediction.

Run via uvicorn:
    uvicorn src.api.main:app --reload

Or as a module (binds 0.0.0.0:8000, no auto-reload):
    python -m src.api.main
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import joblib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.feature_alignment import align_to_training_schema
from src.api.schemas import HealthResponse, PatientEncounter, PredictionResponse, risk_band_for

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_MODEL_PATH = _MODELS_DIR / "lgbm_calibrated.joblib"
_MANIFEST_PATH = _MODELS_DIR / "feature_manifest.json"

# Artifact version string — bump when the model artifact is replaced.
MODEL_VERSION = "lgbm_calibrated_v1"


# ---------------------------------------------------------------------------
# Lifespan: load heavy artifacts once at startup, fail loudly if missing
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and manifest on startup; log shutdown on exit."""
    try:
        logger.info("Loading model artifact: %s", _MODEL_PATH)
        app.state.model = joblib.load(_MODEL_PATH)

        logger.info("Loading feature manifest: %s", _MANIFEST_PATH)
        app.state.manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))

        logger.info(
            "Startup complete — model_version=%s  n_features=%d",
            MODEL_VERSION,
            app.state.manifest["n_features"],
        )
    except Exception:
        # Log the full traceback so the operator sees what went wrong,
        # then re-raise so uvicorn exits with a non-zero code.
        logger.exception("Fatal: failed to load model or manifest — refusing to start")
        raise

    yield

    logger.info("Shutting down MedReadmit API")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedReadmit AI",
    description=(
        "Calibrated 30-day hospital readmission risk prediction. "
        "LightGBM baseline with Platt scaling (AUROC 0.662 on held-out test set)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# TODO: Tighten before Hugging Face Spaces deploy on Day 14.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — last resort for unhandled errors
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected errors: log full traceback, return opaque 500.

    FastAPI's built-in RequestValidationError (422) and HTTPException handlers
    are registered before this generic handler and take priority, so Pydantic
    validation errors still pass through naturally as 422 responses.
    """
    request_id = uuid4().hex
    logger.error(
        "Unhandled exception  request_id=%s  method=%s  path=%s\n%s",
        request_id,
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    # Return an opaque error — never leak the traceback to the client.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and model metadata",
)
async def health(request: Request) -> HealthResponse:
    """Return service status and the feature count the loaded model expects.

    ``status`` is ``"ok"`` when the model is in memory and ready to serve
    predictions; ``"degraded"`` if startup succeeded but the model is missing
    from ``app.state`` (should not happen in normal operation).
    """
    model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
    manifest: dict = getattr(request.app.state, "manifest", {})
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        n_features=manifest.get("n_features", 0),
        model_version=MODEL_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict 30-day readmission risk for one patient encounter",
)
async def predict(encounter: PatientEncounter, request: Request) -> PredictionResponse:
    """Run the full inference pipeline for a single encounter.

    **PHI note**: encounter fields are never written to logs.  Only the
    scalar outputs (probability, risk_band) and timing are logged.
    """
    request_id = uuid4().hex
    t0 = time.perf_counter()

    # model_dump() returns Python attribute names (underscored).  Hyphenated
    # drug-combo fields (e.g. glyburide_metformin) sanitise to the same
    # column names as the training pipeline after apply_feature_transforms.
    aligned = align_to_training_schema(encounter.model_dump(), request.app.state.manifest)

    proba = float(request.app.state.model.predict_proba(aligned)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000
    band = risk_band_for(proba)

    # Structured audit log — encounter fields intentionally omitted (PHI hygiene).
    logger.info(
        "%s",
        json.dumps(
            {
                "request_id": request_id,
                "latency_ms": round(latency_ms, 2),
                "probability": round(proba, 4),
                "risk_band": band,
            }
        ),
    )

    return PredictionResponse(
        probability=proba,
        risk_band=band,
        model_version=MODEL_VERSION,
        request_id=request_id,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
