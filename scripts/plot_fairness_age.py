"""Plot AUROC by age group with 95% bootstrap CIs.

Reads reports/fairness/fairness_age.csv (produced by run_fairness_audit.py
with with_ci=True) and generates a horizontal bar chart with error bars.
Saves to reports/figures/fairness/auroc_by_age.png and logs to MLflow.

Run from the repo root after run_fairness_audit.py has completed:
    python scripts/plot_fairness_age.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FAIRNESS_CSV = Path("reports/fairness/fairness_age.csv")
_OUT_DIR = Path("reports/figures/fairness")
_OUT_FILE = _OUT_DIR / "auroc_by_age.png"


def _load_and_validate(path: Path) -> pd.DataFrame:
    """Load the fairness CSV and check that CI columns are present.

    Args:
        path: Path to fairness_age.csv.

    Returns:
        DataFrame with at least the columns subgroup, n, auroc,
        auroc_ci_low, auroc_ci_high.

    Raises:
        SystemExit: If the file is missing or CI columns are absent
            (re-run run_fairness_audit.py first).
    """
    if not path.exists():
        logger.error("Fairness CSV not found: %s — run run_fairness_audit.py first.", path)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"subgroup", "n", "auroc", "auroc_ci_low", "auroc_ci_high"}
    missing = required - set(df.columns)
    if missing:
        logger.error(
            "Missing columns %s in %s — re-run run_fairness_audit.py with with_ci=True.",
            missing, path,
        )
        sys.exit(1)
    return df


def plot_auroc_by_age(df: pd.DataFrame) -> plt.Figure:
    """Build a horizontal bar chart of AUROC by age group with 95% CI error bars.

    Bars are sorted by AUROC descending.  Sample size is annotated to the right
    of each bar.  NaN CI values are shown without error bars.

    Args:
        df: Fairness DataFrame with columns subgroup, n, auroc,
            auroc_ci_low, auroc_ci_high.

    Returns:
        Matplotlib Figure ready for saving.
    """
    df = df.sort_values("auroc", ascending=True).reset_index(drop=True)

    labels = df["subgroup"].astype(str).tolist()
    aurocs = df["auroc"].values
    ci_low = df["auroc_ci_low"].values
    ci_high = df["auroc_ci_high"].values
    ns = df["n"].values

    # Error bar half-widths; NaN where CI is unavailable.
    xerr_lo = np.where(np.isnan(ci_low), 0.0, aurocs - ci_low)
    xerr_hi = np.where(np.isnan(ci_high), 0.0, ci_high - aurocs)

    fig, ax = plt.subplots(figsize=(8, max(3, 1.2 * len(labels))))

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, aurocs, xerr=[xerr_lo, xerr_hi],
        color="#4c72b0", ecolor="#2d4a7a", capsize=5,
        error_kw={"linewidth": 1.5},
        height=0.55,
    )

    # Annotate sample size to the right of each bar.
    x_right = max(aurocs) + max(xerr_hi) + 0.005
    for i, (n, auroc) in enumerate(zip(ns, aurocs)):
        ax.text(
            x_right, i, f"n={n:,}",
            va="center", ha="left", fontsize=9, color="#555555",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_xlim(0.50, 0.90)
    ax.axvline(0.5, linestyle="--", linewidth=1, color="grey", alpha=0.7, label="random (AUROC = 0.5)")
    ax.set_title(
        "30-day readmission AUROC by age group\n(95% CI via 1000-iter bootstrap)",
        fontsize=13, pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = _load_and_validate(_FAIRNESS_CSV)

    fig = plot_auroc_by_age(df)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", _OUT_FILE)

    # Log to MLflow under the existing fairness_audit experiment.
    mlflow.set_experiment("medreadmit-module1")
    with mlflow.start_run(run_name="fairness_audit_plot"):
        mlflow.log_artifact(str(_OUT_FILE), artifact_path="fairness")
        logger.info("Logged %s to MLflow.", _OUT_FILE.name)


if __name__ == "__main__":
    main()
