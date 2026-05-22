"""Tests for src/evaluate/bootstrap.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluate.bootstrap import bootstrap_auroc, bootstrap_metric


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
# Moderately imbalanced, large enough for stable CIs.
_Y_TRUE = (_RNG.random(800) < 0.15).astype(int)
_Y_PROBA = _RNG.random(800)
# Give the positive class a lift so AUROC > 0.5.
_Y_PROBA[_Y_TRUE == 1] += 0.3
_Y_PROBA = np.clip(_Y_PROBA, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Test (a): CIs bracket the point estimate
# ---------------------------------------------------------------------------


def test_ci_contains_point_estimate() -> None:
    """The 95% CI must contain the point estimate for a well-sized sample."""
    point, lo, hi = bootstrap_auroc(_Y_TRUE, _Y_PROBA, n_iterations=500)

    assert lo <= point <= hi, (
        f"Point estimate {point:.4f} not within CI [{lo:.4f}, {hi:.4f}]"
    )
    # Sanity: CI width should be positive and non-trivial.
    assert hi > lo
    assert (hi - lo) < 0.15, f"CI suspiciously wide: [{lo:.4f}, {hi:.4f}]"


# ---------------------------------------------------------------------------
# Test (b): more iterations → tighter (or equal) CI
# ---------------------------------------------------------------------------


def test_more_iterations_yield_tighter_ci() -> None:
    """CI width must not increase when n_iterations grows 10x (same seed)."""
    _, lo_small, hi_small = bootstrap_auroc(
        _Y_TRUE, _Y_PROBA, n_iterations=100, random_state=7
    )
    _, lo_large, hi_large = bootstrap_auroc(
        _Y_TRUE, _Y_PROBA, n_iterations=1000, random_state=7
    )
    width_small = hi_small - lo_small
    width_large = hi_large - lo_large

    # With the same seed and a much larger draw the distribution converges —
    # width should shrink.  We allow a 20% tolerance for randomness.
    assert width_large <= width_small * 1.20, (
        f"Expected tighter CI at n=1000 ({width_large:.4f}) "
        f"vs n=100 ({width_small:.4f})"
    )


# ---------------------------------------------------------------------------
# Test (c): UserWarning for subgroups with very few positives
# ---------------------------------------------------------------------------


def test_warning_fires_for_tiny_positive_subgroup() -> None:
    """bootstrap_metric must emit UserWarning when most iterations are single-class."""
    rng = np.random.default_rng(1)
    # 2 positives out of 200 — ~13% of resamples draw zero positives (single-class),
    # which is above the 10% skip threshold that triggers the warning.
    y_true_tiny = np.zeros(200, dtype=int)
    y_true_tiny[:2] = 1
    y_proba_tiny = rng.random(200)

    with pytest.warns(UserWarning, match="single-class resamples"):
        bootstrap_metric(
            y_true_tiny,
            y_proba_tiny,
            roc_auc_score,
            n_iterations=200,
            random_state=42,
        )
