"""Construct the binary 30-day readmission label from the raw UCI frame."""

from __future__ import annotations

import logging

import pandas as pd

from src.data.load import load_raw

logger = logging.getLogger(__name__)

# discharge_disposition_id values representing expired or hospice patients.
# These encounters can never result in a readmission, so including them
# suppresses the positive rate and leaks a deterministic 0 label — they
# must be removed before any label is constructed.
_INELIGIBLE_DISPOSITIONS: frozenset[int] = frozenset({11, 13, 14, 19, 20, 21})


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ineligible encounters and create the binary readmission label.

    Removes expired and hospice discharges, then derives ``readmitted_30d``
    from the raw ``readmitted`` column before dropping it. No other columns
    are added or removed.

    Args:
        df: Raw DataFrame from :func:`src.data.load.load_raw`. Must contain
            ``discharge_disposition_id`` and ``readmitted`` columns.

    Returns:
        DataFrame with ineligible rows removed and a new integer column
        ``readmitted_30d`` (1 = readmitted within 30 days, 0 = otherwise).
        The original ``readmitted`` column is dropped.
    """
    logger.info("build_target: starting with %d rows", len(df))

    ineligible = df["discharge_disposition_id"].isin(_INELIGIBLE_DISPOSITIONS)
    df = df[~ineligible].copy()
    logger.info(
        "Dropped %d expired/hospice encounters — %d rows remain",
        int(ineligible.sum()),
        len(df),
    )

    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    logger.info("Positive-class rate (readmitted_30d=1): %.4f", df["readmitted_30d"].mean())
    return df


if __name__ == "__main__":
    # Run from the repo root: python -m src.data.make_target
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")
    df = load_raw()
    df = build_target(df)
    print(df.shape, df["readmitted_30d"].mean())
