"""Download the UCI Diabetes 130-US Hospitals dataset to data/raw/."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

logger = logging.getLogger(__name__)

# Resolved relative to this file so the script works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _REPO_ROOT / "data" / "raw"
DATASET_ID = 296  # https://archive.ics.uci.edu/dataset/296/


def download(dest_dir: Path = RAW_DIR) -> Path:
    """Fetch the UCI Diabetes dataset and write it to a CSV file.

    Args:
        dest_dir: Directory to write ``diabetes.csv``. Created if absent.

    Returns:
        Absolute path of the saved CSV.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "diabetes.csv"

    if out_path.exists():
        logger.info("Already exists, skipping download: %s", out_path)
        return out_path

    logger.info("Fetching dataset %d from UCI ML Repository…", DATASET_ID)
    dataset = fetch_ucirepo(id=DATASET_ID)

    # dataset.data.original is the full, unmodified table exactly as UCI serves
    # it — features + target column in one frame. Prefer it over pd.concat of
    # dataset.data.features + dataset.data.targets because the latter can
    # duplicate the index when a patient has multiple encounters.
    df: pd.DataFrame = dataset.data.original

    df.to_csv(out_path, index=False)
    logger.info(
        "Saved %d rows × %d cols → %s",
        len(df),
        df.shape[1],
        out_path,
    )
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    path = download()
    print(path)
