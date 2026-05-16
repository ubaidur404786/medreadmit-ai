"""Reusable binary-classification evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute AUROC, AUPRC, Brier score, and the best-F1 threshold.

    The best-F1 threshold is found by sweeping the precision-recall curve and
    picking the operating point that maximises the harmonic mean of precision
    and recall.  This is more meaningful than a fixed 0.5 cut-off for
    imbalanced datasets like this one (~11 % positive rate).

    Args:
        y_true: 1-D array of 0/1 ground-truth labels.
        y_proba: 1-D array of predicted probabilities for class 1.
        prefix: Optional string prepended to every key (e.g. ``"val"`` →
            ``"val_auroc"``).  An underscore separator is added automatically.

    Returns:
        Dictionary with keys ``{p}auroc``, ``{p}auprc``, ``{p}brier``,
        ``{p}best_f1``, ``{p}best_threshold`` where ``{p}`` is ``prefix + "_"``
        when *prefix* is non-empty, else empty string.
    """
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len-n arrays for precision/recall but a
    # len-(n-1) array for thresholds; the final point is (1.0, 0.0) with no
    # corresponding threshold, so we slice to align.
    pr, re = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where((pr + re) > 0, 2 * pr * re / (pr + re + 1e-12), 0.0)
    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    best_threshold = float(thresholds[best_idx])

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}auroc": float(auroc),
        f"{p}auprc": float(auprc),
        f"{p}brier": float(brier),
        f"{p}best_f1": best_f1,
        f"{p}best_threshold": best_threshold,
    }


def calibration_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Path | None = None,
) -> Figure:
    """Produce a reliability diagram (calibration curve).

    Bins predicted probabilities and compares the mean predicted probability
    in each bin against the observed positive rate.  A perfectly calibrated
    model lies on the diagonal.

    Args:
        y_true: 1-D array of 0/1 ground-truth labels.
        y_proba: 1-D array of predicted probabilities for class 1.
        n_bins: Number of equal-width bins for the calibration curve.
        save_path: If provided, the figure is saved to this path before
            being returned.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure` object.
    """
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve (reliability diagram)")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def confusion_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute confusion-matrix statistics at a fixed probability threshold.

    Args:
        y_true: 1-D array of 0/1 ground-truth labels.
        y_proba: 1-D array of predicted probabilities for class 1.
        threshold: Decision boundary; predictions >= threshold are class 1.

    Returns:
        Dictionary with keys ``tp``, ``fp``, ``tn``, ``fn``, ``precision``,
        ``recall``, ``specificity``.  Rate metrics are 0.0 when the
        denominator is zero.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }
