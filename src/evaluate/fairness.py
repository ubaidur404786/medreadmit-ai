"""Subgroup fairness audit utilities for the readmission model."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate.bootstrap import bootstrap_auroc, bootstrap_auprc

logger = logging.getLogger(__name__)


def subgroup_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    subgroup: pd.Series,
    min_samples: int = 50,
    with_ci: bool = False,
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
        with_ci: If True, add 95% bootstrap CI columns for AUROC and AUPRC
            (auroc_ci_low, auroc_ci_high, auprc_ci_low, auprc_ci_high).
            Subgroups where >10% of bootstrap iterations are skipped get
            np.nan CI cells. Default False preserves backward compatibility.

    Returns:
        DataFrame with columns ``subgroup``, ``n``, ``prevalence``,
        ``mean_pred``, ``auroc``, ``auprc``, sorted by ``n`` descending.
        When ``with_ci=True``, four additional CI columns are appended.
        All numeric columns are rounded to 3 decimal places (CIs) or 4 (others).
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
                "Skipping subgroup %r — only %d positive labels (need >= 5 for stable AUROC)",
                group_val,
                n_pos,
            )
            continue

        row: dict = {
            "subgroup": group_val,
            "n": n,
            "prevalence": round(float(yt.mean()), 4),
            "mean_pred": round(float(yp.mean()), 4),
            "auroc": round(float(roc_auc_score(yt, yp)), 4),
            "auprc": round(float(average_precision_score(yt, yp)), 4),
        }

        if with_ci:
            # Catch UserWarning from bootstrap when >10% of iters are skipped.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, auroc_lo, auroc_hi = bootstrap_auroc(yt, yp)
                _, auprc_lo, auprc_hi = bootstrap_auprc(yt, yp)

            if caught:
                logger.warning(
                    "Subgroup %r: bootstrap CIs unreliable — setting to NaN. (%s)",
                    group_val,
                    caught[0].message,
                )
                auroc_lo = auroc_hi = auprc_lo = auprc_hi = float("nan")

            row["auroc_ci_low"] = round(auroc_lo, 3)
            row["auroc_ci_high"] = round(auroc_hi, 3)
            row["auprc_ci_low"] = round(auprc_lo, 3)
            row["auprc_ci_high"] = round(auprc_hi, 3)

        rows.append(row)

    base_cols = ["subgroup", "n", "prevalence", "mean_pred", "auroc", "auprc"]
    ci_cols = ["auroc_ci_low", "auroc_ci_high", "auprc_ci_low", "auprc_ci_high"]

    if not rows:
        logger.warning("No subgroups passed the minimum-sample filter.")
        all_cols = base_cols + ci_cols if with_ci else base_cols
        return pd.DataFrame(columns=all_cols)

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def format_fairness_table(df: pd.DataFrame, group_name: str) -> str:
    """Return a pretty-printed string of the subgroup metrics table.

    When the DataFrame contains CI columns (auroc_ci_low / auroc_ci_high),
    AUROC and AUPRC are printed as ``0.671 [0.654, 0.687]``.

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

    has_ci = "auroc_ci_low" in df.columns

    title = f"Fairness audit — {group_name}"

    if has_ci:
        header = (
            f"  {'Subgroup':<28}  {'n':>6}  {'Prev':>6}  "
            f"{'MeanPred':>8}  {'AUROC [95% CI]':<22}  {'AUPRC [95% CI]':<22}"
        )
    else:
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
        if has_ci:
            auroc_lo = row.get("auroc_ci_low")
            auroc_hi = row.get("auroc_ci_high")
            auprc_lo = row.get("auprc_ci_low")
            auprc_hi = row.get("auprc_ci_high")

            if pd.isna(auroc_lo):
                auroc_str = f"{row['auroc']:.3f} [n/a]"
            else:
                auroc_str = f"{row['auroc']:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]"

            if pd.isna(auprc_lo):
                auprc_str = f"{row['auprc']:.3f} [n/a]"
            else:
                auprc_str = f"{row['auprc']:.3f} [{auprc_lo:.3f}, {auprc_hi:.3f}]"

            lines.append(
                f"  {str(row['subgroup']):<28}  {int(row['n']):>6}  "
                f"{row['prevalence']:>6.4f}  {row['mean_pred']:>8.4f}  "
                f"{auroc_str:<22}  {auprc_str:<22}"
            )
        else:
            lines.append(
                f"  {str(row['subgroup']):<28}  {int(row['n']):>6}  "
                f"{row['prevalence']:>6.4f}  {row['mean_pred']:>8.4f}  "
                f"{row['auroc']:>6.4f}  {row['auprc']:>6.4f}"
            )

    lines.append("=" * width)
    lines.append("")

    return "\n".join(lines)
