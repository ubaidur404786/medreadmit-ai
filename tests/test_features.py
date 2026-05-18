"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import re

import pytest

from src.data.load import load_raw
from src.data.make_target import build_target
from src.features.build_features import build_features

# Characters that LightGBM (and many other frameworks) reject in column names.
_BAD_CHARS = re.compile(r"[\[\],\"<>:{}=]")


@pytest.fixture(scope="module")
def feature_data():
    """Load 1000 real rows once per module and run build_features."""
    df = build_target(load_raw())
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    return build_features(df)


def test_no_leakage_columns(feature_data) -> None:
    X, y, groups = feature_data
    assert "patient_nbr" not in X.columns
    assert "readmitted_30d" not in X.columns


def test_sanitized_feature_names(feature_data) -> None:
    X, _, _ = feature_data
    bad = [col for col in X.columns if _BAD_CHARS.search(col)]
    assert bad == [], f"Columns with disallowed characters: {bad}"


def test_shape_consistency(feature_data) -> None:
    X, y, groups = feature_data
    assert len(X) == len(y) == len(groups)
