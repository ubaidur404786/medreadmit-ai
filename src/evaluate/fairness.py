"""Subgroup fairness audit utilities for the readmission model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def subgroup_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    subgroup: pd.Series,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Compute AUROC, AUPRC, prevalence, and mean predicted risk per subgroup.

    Subgroups that are too small or too imbalanced for stable metric estimation
    are dropped and logged as warnings rather than silently omitted.

    Args:
        y_true: 1-D array of 0/1 ground-truth labels, aligned with ``subgroup``.
        y_proba: 1-D array of predicted probabilities for class 1.
        subgroup: Categorical Series (e.g. race, age group) with the same
            length as ``y_true``.
        min_samples: Subgroups with fewer rows than this threshold are skipped.
            Default 50 is a practical lower bound for stable AUROC estimation.

    Returns:
        DataFrame with columns ``subgroup``, ``n``, ``prevalence``,
        ``mean_pred``, ``auroc``, ``auprc``, sorted by ``n`` descending.
        All numeric columns are rounded to 4 decimal places.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rows: list[dict] = []

    for group_val in subgroup.unique():
        mask = (subgroup == group_val).values
        yt = y_true[mask]
        yp = y_proba[mask]
        n = int(mask.sum())

        if n < min_samples:
            logger.warning(
                "Skipping subgroup %r — only %d rows (min_samples=%d)",
                group_val,
                n,
                min_samples,
            )
            continue

        n_pos = int(yt.sum())
        if n_pos < 5:
            logger.warning(
                "Skipping subgroup %r — only %d positive labels (need ≥ 5 for stable AUROC)",
                group_val,
                n_pos,
            )
            continue

        rows.append(
            {
                "subgroup": group_val,
                "n": n,
                "prevalence": round(float(yt.mean()), 4),
                "mean_pred": round(float(yp.mean()), 4),
                "auroc": round(float(roc_auc_score(yt, yp)), 4),
                "auprc": round(float(average_precision_score(yt, yp)), 4),
            }
        )

    if not rows:
        logger.warning("No subgroups passed the minimum-sample filter.")
        return pd.DataFrame(columns=["subgroup", "n", "prevalence", "mean_pred", "auroc", "auprc"])

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def format_fairness_table(df: pd.DataFrame, group_name: str) -> str:
    """Return a pretty-printed string of the subgroup metrics table.

    Args:
        df: Output of :func:`subgroup_metrics`.
        group_name: Human-readable name of the grouping variable (e.g.
            ``"race"``, ``"age"``) used as the table title.

    Returns:
        Formatted multi-line string suitable for printing to stdout or
        embedding in a report.
    """
    if df.empty:
        return f"Fairness audit — {group_name}: no subgroups to display.\n"

    title = f"Fairness audit — {group_name}"
    header = (
        f"  {'Subgroup':<28}  {'n':>6}  {'Prev':>6}  "
        f"{'MeanPred':>8}  {'AUROC':>6}  {'AUPRC':>6}"
    )
    divider = "  " + "-" * (len(header) - 2)
    width = max(len(title), len(header))

    lines = [
        f"\n{title:^{width}}",
        "=" * width,
        header,
        divider,
    ]
    for _, row in df.iterrows():
        lines.append(
            f"  {str(row['subgroup']):<28}  {int(row['n']):>6}  "
            f"{row['prevalence']:>6.4f}  {row['mean_pred']:>8.4f}  "
            f"{row['auroc']:>6.4f}  {row['auprc']:>6.4f}"
        )
    lines.append("=" * width)
    lines.append("")

    return "\n".join(lines)
