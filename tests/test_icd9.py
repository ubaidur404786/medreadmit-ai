"""Tests for ICD-9 code grouping logic."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.icd9_grouping import map_icd9_to_group


def test_circulatory_range() -> None:
    codes = ["390", "410.5", "459", "785"]
    for code in codes:
        assert map_icd9_to_group(code) == "circulatory", f"{code!r} should be circulatory"


def test_diabetes_overrides_other() -> None:
    # "250.83" starts with "250" — the string-based diabetes rule fires
    # before any numeric range check, so it must not fall through to "other".
    assert map_icd9_to_group("250.83") == "diabetes"


def test_v_and_e_codes_map_to_other() -> None:
    codes = ["V57", "V58.1", "E906"]
    for code in codes:
        assert map_icd9_to_group(code) == "other", f"{code!r} should be other"


def test_nan_and_empty_map_to_missing() -> None:
    for value in [np.nan, "", None]:
        assert map_icd9_to_group(value) == "missing", f"{value!r} should be missing"
