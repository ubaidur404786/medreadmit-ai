"""Load the raw UCI Diabetes CSV into a DataFrame."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Diagnosis columns contain mixed ICD-9 strings ("250.00", "V58", "E887").
# Without an explicit dtype, a column with only numeric-looking codes would be
# inferred as float — "250.00" becomes 250.0, losing the sub-code precision.
# skipinitialspace (below) strips the leading space from the raw header token
# " diag_1" → "diag_1" so these names match the dtype dict at parse time.
_DIAG_DTYPES: dict[str, type] = {"diag_1": str, "diag_2": str, "diag_3": str}

# At the top of the file, after the imports:
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/data/load.py → repo root
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"


def load_raw(path: Path = DEFAULT_RAW_CSV) -> pd.DataFrame:
    """Read the raw UCI Diabetes CSV and normalise whitespace-encoded missings.

    The CSV written by ``ucimlrepo`` inherits a fixed-width layout from the
    original UCI source: every column name and cell value is space-padded.
    Missing entries are whitespace-only strings, not the ``"?"`` sentinel
    mentioned in the dataset README. This function strips that padding and
    converts blank cells to ``np.nan``.

    Does no filtering, target construction, or feature engineering.

    Args:
        path: Path to the CSV file. Resolved relative to the caller's working
            directory; invoke from the repo root or pass an absolute path.

    Returns:
        DataFrame with 50 columns and ~101 k rows. Diagnosis columns retain
        their ``object`` dtype. All other dtypes are pandas defaults.
    """
    logger.info("Reading %s", path)
    df = pd.read_csv(
        path,
        dtype=_DIAG_DTYPES,
        skipinitialspace=True,  # strips leading whitespace from every token
        low_memory=False,
    )
    # skipinitialspace only strips leading whitespace; some column names
    # still carry trailing padding (e.g. "race           ").
    df.columns = df.columns.str.strip()
    # Strip remaining trailing whitespace from object-column values and map
    # any now-blank strings to NaN.
    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip()).replace("", np.nan)
    logger.info(
        "Loaded %d rows × %d cols — %d NaN values total",
        len(df),
        df.shape[1],
        int(df.isna().sum().sum()),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frame = load_raw()
    print(f"\nShape: {frame.shape}")
    top_missing = frame.isna().mean().sort_values(ascending=False).head(10)
    print("\nTop-10 most-missing columns (fraction NaN):")
    print(top_missing.to_string())
