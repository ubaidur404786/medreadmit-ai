"""Tests for patient-grouped train/val/test splitting."""

from __future__ import annotations

import pytest

from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import patient_grouped_split
from src.features.build_features import build_features


@pytest.fixture(scope="module")
def split_data():
    """Load 1000 real rows once per module, build features, and split."""
    df = build_target(load_raw())
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    X, y, groups = build_features(df)
    splits = patient_grouped_split(X, y, groups)
    return (*splits, groups)  # X_train, X_val, X_test, y_train, y_val, y_test, groups


def test_split_sizes_sum_to_total(split_data) -> None:
    X_train, X_val, X_test, *_ = split_data
    total = len(X_train) + len(X_val) + len(X_test)
    # GroupShuffleSplit assigns every row to exactly one split.
    assert total == 1000


def test_no_patient_overlap(split_data) -> None:
    X_train, X_val, X_test, y_train, y_val, y_test, groups = split_data
    train_pts = set(groups[X_train.index])
    val_pts = set(groups[X_val.index])
    test_pts = set(groups[X_test.index])
    assert not (train_pts & val_pts), "Patients leaked between train and val"
    assert not (train_pts & test_pts), "Patients leaked between train and test"
    assert not (val_pts & test_pts), "Patients leaked between val and test"
