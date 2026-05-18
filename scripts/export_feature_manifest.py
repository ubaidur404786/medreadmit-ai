"""Export a feature manifest from the production model and current feature pipeline.

Runs the full data pipeline to materialise the training-time feature matrix,
loads the calibrated model, validates that feature names agree, then writes
models/feature_manifest.json as a stable contract for the inference layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.data.load import load_raw
from src.data.make_target import build_target
from src.features.build_features import build_features
from src.features.icd9_grouping import bucket_diagnosis_columns  # noqa: F401 (imported for clarity)

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MANIFEST_PATH = MODELS_DIR / "feature_manifest.json"


def _extract_lgbm_feature_names(model: object) -> list[str]:
    """Walk the model object to find LightGBM feature names.

    PlattWrapper stores the base LGBMClassifier under .base_model; the fitted
    LightGBM booster exposes feature names via .feature_name_ (scikit-learn API)
    or .booster_.feature_name() (native booster API).  We try each layer in turn
    and raise a clear error if nothing works rather than silently returning None.

    Args:
        model: The loaded joblib artifact (may be PlattWrapper or bare LGBMClassifier).

    Returns:
        Ordered list of feature name strings as seen during training.
    """
    candidates = [model]

    # Unwrap PlattWrapper
    if hasattr(model, "base_model"):
        logger.info("Detected PlattWrapper — unwrapping .base_model")
        candidates.insert(0, model.base_model)

    # Also try common sklearn wrapper attributes in case of future nesting.
    for attr in ("estimator_", "base_estimator_"):
        if hasattr(model, attr):
            candidates.insert(0, getattr(model, attr))

    for candidate in candidates:
        # Preferred: sklearn-style fitted attribute
        if hasattr(candidate, "feature_name_"):
            names = list(candidate.feature_name_)
            logger.info("Feature names sourced from .feature_name_ (%d features)", len(names))
            return names
        # Fallback: native LightGBM booster
        if hasattr(candidate, "booster_") and hasattr(candidate.booster_, "feature_name"):
            names = candidate.booster_.feature_name()
            logger.info(
                "Feature names sourced from .booster_.feature_name() (%d features)", len(names)
            )
            return names

    raise AttributeError(
        "Could not extract feature names from model artifact.  "
        "Expected a PlattWrapper wrapping an LGBMClassifier, or a bare LGBMClassifier.  "
        f"Inspected attributes on: {[type(c).__name__ for c in candidates]}"
    )


def main() -> None:
    """Run the manifest export pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Materialise the feature matrix exactly as training did
    # ------------------------------------------------------------------
    logger.info("Loading raw data…")
    raw = load_raw()
    logger.info("Building target…")
    labelled = build_target(raw)
    logger.info("Building features…")
    X, y, _groups = build_features(labelled)
    logger.info("Feature matrix ready: %d rows × %d columns", *X.shape)

    # ------------------------------------------------------------------
    # 2. Load model and extract expected feature names
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / "lgbm_calibrated.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    model_feature_names = _extract_lgbm_feature_names(model)

    # ------------------------------------------------------------------
    # 3. Assert feature name agreement — fail loudly on drift
    # ------------------------------------------------------------------
    pipeline_cols = X.columns.tolist()

    if model_feature_names != pipeline_cols:
        model_set = set(model_feature_names)
        pipeline_set = set(pipeline_cols)
        only_in_model = sorted(model_set - pipeline_set)
        only_in_pipeline = sorted(pipeline_set - model_set)
        in_both_wrong_order = (
            model_set == pipeline_set and model_feature_names != pipeline_cols
        )
        detail_lines = []
        if only_in_model:
            detail_lines.append(f"  In model but not pipeline ({len(only_in_model)}): {only_in_model[:10]}")
        if only_in_pipeline:
            detail_lines.append(
                f"  In pipeline but not model ({len(only_in_pipeline)}): {only_in_pipeline[:10]}"
            )
        if in_both_wrong_order:
            detail_lines.append("  Column sets match but ORDER differs — check pd.get_dummies sort stability")
        raise AssertionError(
            "Feature drift detected: model and pipeline feature names disagree.\n"
            + "\n".join(detail_lines)
        )

    logger.info("Feature name validation passed — %d features align exactly", len(model_feature_names))

    # ------------------------------------------------------------------
    # 4. Compute manifest fields
    # ------------------------------------------------------------------
    feature_dtypes = {col: str(dtype) for col, dtype in X.dtypes.items()}
    positive_rate = round(float(y.mean()), 4)

    sorted_cols_sha256 = hashlib.sha256(
        json.dumps(sorted(model_feature_names)).encode()
    ).hexdigest()

    manifest = {
        "manifest_version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(model_feature_names),
        "feature_columns": model_feature_names,
        "feature_dtypes": feature_dtypes,
        "positive_rate_train": positive_rate,
    }

    # ------------------------------------------------------------------
    # 5. Write (idempotent overwrite)
    # ------------------------------------------------------------------
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest written → %s", MANIFEST_PATH)

    # ------------------------------------------------------------------
    # 6. Summary log
    # ------------------------------------------------------------------
    logger.info("n_features: %d", manifest["n_features"])
    logger.info("positive_rate_train: %.4f", manifest["positive_rate_train"])
    logger.info("sha256(sorted columns): %s", sorted_cols_sha256)


if __name__ == "__main__":
    main()
