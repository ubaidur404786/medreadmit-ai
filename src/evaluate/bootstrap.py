"""Percentile bootstrap confidence intervals for binary classification metrics."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_metric(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a metric_fn taking (y_true, y_proba).

    Resamples rows with replacement n_iterations times.  Resamples that yield
    only one class are skipped because the metric is undefined.  If more than
    10% of iterations are skipped, a UserWarning is raised — the subgroup is
    too small or too imbalanced for reliable CIs.

    Args:
        y_true: 1-D array of 0/1 ground-truth labels.
        y_proba: 1-D array of predicted probabilities for class 1.
        metric_fn: Callable with signature ``(y_true, y_proba) -> float``.
        n_iterations: Number of bootstrap resamples (default 1000).
        ci: Confidence level, e.g. 0.95 for a 95% CI (default 0.95).
        random_state: Seed for the NumPy RNG (default 42).

    Returns:
        Tuple of ``(point_estimate, ci_lower, ci_upper)``.

    Raises:
        UserWarning: If more than 10% of iterations are skipped due to
            single-class resamples.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_proba)

    rng = np.random.default_rng(random_state)
    boot_scores: list[float] = []
    skipped = 0

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt_boot = y_true[idx]
        yp_boot = y_proba[idx]
        if len(np.unique(yt_boot)) < 2:
            skipped += 1
            continue
        boot_scores.append(metric_fn(yt_boot, yp_boot))

    skip_rate = skipped / n_iterations
    if skip_rate > 0.10:
        warnings.warn(
            f"Bootstrap skipped {skipped}/{n_iterations} iterations ({skip_rate:.1%}) "
            "due to single-class resamples. CIs are unreliable — subgroup may be too "
            "small or too imbalanced.",
            UserWarning,
            stacklevel=2,
        )

    if not boot_scores:
        alpha = (1.0 - ci) / 2.0
        return point_estimate, float("nan"), float("nan")

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_scores, 100 * alpha))
    ci_upper = float(np.percentile(boot_scores, 100 * (1.0 - alpha)))
    return point_estimate, ci_lower, ci_upper


def bootstrap_auroc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    **kwargs: object,
) -> tuple[float, float, float]:
    """Convenience wrapper: bootstrap_metric with roc_auc_score.

    Args:
        y_true: 1-D array of 0/1 ground-truth labels.
        y_proba: 1-D array of predicted probabilities for class 1.
        **kwargs: Forwarded to :func:`bootstrap_metric`
            (n_iterations, ci, random_state).

    Returns:
        Tuple of ``(auroc, ci_lower, ci_upper)``.
    """
    return bootstrap_metric(y_true, y_proba, roc_auc_score, **kwargs)


def bootstrap_auprc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    **kwargs: object,
) -> tuple[float, float, float]:
    """Convenience wrapper: bootstrap_metric with average_precision_score.

    Args:
        y_true: 1-D array of 0/1 ground-truth labels.
        y_proba: 1-D array of predicted probabilities for class 1.
        **kwargs: Forwarded to :func:`bootstrap_metric`
            (n_iterations, ci, random_state).

    Returns:
        Tuple of ``(auprc, ci_lower, ci_upper)``.
    """
    return bootstrap_metric(y_true, y_proba, average_precision_score, **kwargs)
