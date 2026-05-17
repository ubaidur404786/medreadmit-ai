"""Tests for target construction and ineligible-encounter filtering."""

from __future__ import annotations

import pandas as pd

from src.data.make_target import build_target

# Ineligible disposition ids per _INELIGIBLE_DISPOSITIONS in make_target.py:
# {11, 13, 14, 19, 20, 21}


def test_expired_dropped() -> None:
    df = pd.DataFrame(
        {
            "discharge_disposition_id": [1, 6, 11, 13, 19, 25],
            "readmitted": ["<30", "NO", "<30", "NO", ">30", "<30"],
        }
    )
    result = build_target(df)
    # Rows with disposition 11, 13, 19 must be removed.
    assert set(result["discharge_disposition_id"]) == {1, 6, 25}
    assert len(result) == 3


def test_target_is_binary_and_correct() -> None:
    df = pd.DataFrame(
        {
            "discharge_disposition_id": [1, 1, 1],
            "readmitted": ["<30", ">30", "NO"],
        }
    )
    result = build_target(df)
    assert list(result["readmitted_30d"]) == [1, 0, 0]
    assert "readmitted" not in result.columns
