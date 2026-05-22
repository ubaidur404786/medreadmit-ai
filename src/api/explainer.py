"""API-friendly SHAP explainer for per-request local feature attribution.

Design constraints
------------------
* ``shap.TreeExplainer`` construction is expensive (~1–2 s per model).  The
  :class:`APIExplainer` is built **once** at app startup and stored on
  ``app.state``; individual ``explain()`` calls take ~10 ms.
* The explainer must wrap the *underlying LightGBM estimator*, not the
  :class:`~src.models.platt_wrapper.PlattWrapper`.  SHAP's ``TreeExplainer``
  only supports tree-native objects.
* SHAP values live in **log-odds space** (raw LightGBM, before Platt scaling).
  Platt scaling shifts the *level* of predicted probabilities but preserves
  feature *rank order*, so the attribution is still valid for the calibrated
  model.  Consumers of :meth:`APIExplainer.explain` should surface this caveat
  in UI copy: "Feature contributions reflect the uncalibrated model score."
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# Attribute names to probe when unwrapping a model to reach the underlying
# LightGBM estimator, tried in this order:
#   1. base_model   — PlattWrapper's attribute (our production wrapper)
#   2. estimator_   — sklearn Pipeline / CalibratedClassifierCV (fitted attr)
#   3. base_estimator_ — older sklearn meta-estimator convention
#   4. model_       — some custom wrappers
_UNWRAP_ATTRS = ("base_model", "estimator_", "base_estimator_", "model_")


def _extract_lgbm(model: object) -> object:
    """Unwrap a model wrapper to find the underlying tree estimator.

    Tries each attribute in :data:`_UNWRAP_ATTRS` in order and returns the
    first one whose extracted object exposes ``predict_proba``.  If nothing
    matches, raises a :class:`TypeError` with a diagnostic message that names
    the wrapper type and all probed attributes.

    Args:
        model: The top-level model object (e.g. a loaded :class:`PlattWrapper`
            or a bare :class:`~lightgbm.LGBMClassifier`).

    Returns:
        The innermost tree estimator suitable for :class:`shap.TreeExplainer`.
    """
    # Fast path: the model itself is already a tree estimator.
    if _looks_like_tree_model(model) and not _is_wrapper(model):
        return model

    for attr in _UNWRAP_ATTRS:
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if _looks_like_tree_model(candidate):
                logger.debug(
                    "Unwrapped %s via .%s → %s",
                    type(model).__name__,
                    attr,
                    type(candidate).__name__,
                )
                return candidate

    raise TypeError(
        f"Cannot extract a tree estimator from {type(model).__name__!r}. "
        f"Tried attributes: {_UNWRAP_ATTRS}. "
        "The object must be a PlattWrapper (with .base_model) or an sklearn "
        "meta-estimator wrapping a tree model. "
        f"Available attributes: {[a for a in dir(model) if not a.startswith('__')]}"
    )


def _looks_like_tree_model(obj: object) -> bool:
    """Heuristic: does this object look like a tree-based estimator?"""
    # shap.TreeExplainer accepts objects with a .predict method or a known
    # type string.  LGBMClassifier has both predict_proba and booster_.
    return hasattr(obj, "predict_proba") and (
        hasattr(obj, "booster_") or hasattr(obj, "get_booster")
    )


def _is_wrapper(obj: object) -> bool:
    """True if the object is a wrapper (has one of the unwrap attributes)."""
    return any(hasattr(obj, a) for a in _UNWRAP_ATTRS)


def _normalise_shap(raw: Any, n_samples: int, n_features: int) -> np.ndarray:
    """Collapse SHAP output to a 2-D (n_samples, n_features) array for class 1.

    Mirrors the logic in :func:`src.explain.shap_utils.compute_shap_values`
    to handle API variation across SHAP versions:

    * ``list`` of two arrays — older SHAP: ``[class-0, class-1]``; take ``[1]``.
    * ``ndarray`` with ``ndim == 3`` — shape ``(n, f, 2)``; slice ``[:, :, 1]``.
    * ``ndarray`` with ``ndim == 2`` — already ``(n, f)`` for the positive class.
    """
    if isinstance(raw, list):
        arr = np.asarray(raw[1])
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        arr = raw[:, :, 1]
    else:
        arr = np.asarray(raw)

    if arr.shape != (n_samples, n_features):
        raise ValueError(
            f"Unexpected SHAP array shape {arr.shape}; "
            f"expected ({n_samples}, {n_features})"
        )
    return arr


class APIExplainer:
    """Pre-built SHAP TreeExplainer for fast per-request local explanations.

    Construct once at application startup (stored on ``app.state.explainer``)
    and reuse across all requests.  Per-request calls to :meth:`explain` do
    *not* rebuild the explainer.

    **SHAP values are in log-odds space** (raw LightGBM, before Platt scaling).
    Platt scaling re-calibrates predicted probabilities but does not reorder
    feature contributions — the feature ranking is valid for the calibrated
    model.  Surface this to end-users: "Contributions shown in log-odds units;
    relative ranking is preserved after probability calibration."
    """

    def __init__(self, model: object, feature_names: list[str]) -> None:
        """Build the TreeExplainer around the underlying LightGBM estimator.

        Args:
            model: Top-level model artifact (typically a
                :class:`~src.models.platt_wrapper.PlattWrapper`).  The
                LightGBM estimator is extracted automatically.
            feature_names: Ordered list of feature names as seen at training
                time (``manifest["feature_columns"]``).

        Raises:
            TypeError: If no tree estimator can be extracted from *model*.
        """
        lgbm_estimator = _extract_lgbm(model)
        logger.info(
            "Building TreeExplainer on %s", type(lgbm_estimator).__name__
        )
        self._explainer: shap.TreeExplainer = shap.TreeExplainer(lgbm_estimator)
        self._feature_names: list[str] = list(feature_names)

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Return the top-k SHAP contributions for each row in *X*.

        SHAP values represent each feature's additive contribution to the
        model's log-odds output (pre-Platt).  Values are normalised to the
        positive class (readmitted within 30 days).

        Args:
            X: Feature DataFrame, typically of shape ``(1, n_features)`` for
                per-request use.  Must have columns in the same order as the
                ``feature_names`` passed to the constructor.
            top_k: Number of top features to return per row, ranked by
                descending ``|shap_value|``.

        Returns:
            List of length ``len(X)``.  Each element is a ``list`` of *top_k*
            dicts, sorted by descending ``|shap_value|``::

                [
                    {"feature": "number_inpatient",
                     "shap_value": 0.412,
                     "feature_value": 2.0},
                    ...
                ]

            All numbers are plain Python :class:`float` (JSON-serialisable).
        """
        raw = self._explainer.shap_values(X)
        shap_arr = _normalise_shap(raw, n_samples=len(X), n_features=X.shape[1])

        # Convert to numpy once; avoids N×top_k pandas iloc/label-lookup calls.
        X_arr = X.to_numpy()  # shape: (n_samples, n_features)

        results: list[list[dict[str, Any]]] = []
        for row_idx in range(len(X)):
            row_shap = shap_arr[row_idx]  # shape: (n_features,)
            # argsort ascending, reverse for descending |value|, take top_k
            top_indices = np.argsort(np.abs(row_shap))[::-1][:top_k]
            contributions = [
                {
                    "feature": self._feature_names[i],
                    "shap_value": float(row_shap[i]),
                    "feature_value": float(X_arr[row_idx, i]),
                }
                for i in top_indices
            ]
            results.append(contributions)

        return results
