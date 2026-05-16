"""Patient-grouped train / val / test splitting for the UCI Diabetes dataset."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def patient_grouped_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split encounters into train / val / test with no patient appearing in more than one split.

    Uses a two-stage :class:`~sklearn.model_selection.GroupShuffleSplit`:
    first carves out the test set, then splits the remainder into train and val.
    Group membership (``patient_nbr``) is respected at every stage so that all
    encounters for a given patient land in exactly one split.

    Args:
        X: Feature matrix, as returned by :func:`src.features.build_features`.
        y: Binary target series (``readmitted_30d``).
        groups: Patient identifier array, same length as ``X``.
        test_size: Fraction of encounters to reserve for the test set.
        val_size: Fraction of *total* encounters to reserve for validation.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Stage 1: hive off the held-out test set.
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_outer.split(X, y, groups))

    X_trainval, y_trainval = X.iloc[trainval_idx], y.iloc[trainval_idx]
    groups_trainval = groups[trainval_idx]

    logger.info(
        "Stage 1: %d trainval rows, %d test rows (%d unique patients each)",
        len(trainval_idx),
        len(test_idx),
        len(set(groups_trainval)),
        # groups_test patient count is logged at the end
    )

    # Stage 2: split trainval into train and val.
    # val_size is expressed as a fraction of the *total* dataset, so we rescale:
    #   inner_val_frac = val_size / (1 - test_size)
    inner_val_size = val_size / (1.0 - test_size)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=inner_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_inner.split(X_trainval, y_trainval, groups_trainval))

    X_train = X_trainval.iloc[train_idx]
    X_val = X_trainval.iloc[val_idx]
    X_test = X.iloc[test_idx]

    y_train = y_trainval.iloc[train_idx]
    y_val = y_trainval.iloc[val_idx]
    y_test = y.iloc[test_idx]

    logger.info(
        "Final split sizes — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    logger.info(
        "Positive-class rate — train: %.4f, val: %.4f, test: %.4f",
        y_train.mean(),
        y_val.mean(),
        y_test.mean(),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def assert_no_patient_leakage(
    groups_train: np.ndarray,
    groups_val: np.ndarray,
    groups_test: np.ndarray,
) -> None:
    """Raise AssertionError if any patient appears in more than one split.

    Checks all three pairwise intersections using set arithmetic.

    Args:
        groups_train: Patient identifiers in the training split.
        groups_val: Patient identifiers in the validation split.
        groups_test: Patient identifiers in the test split.

    Raises:
        AssertionError: If train ∩ val, train ∩ test, or val ∩ test is non-empty.
    """
    train_set = set(groups_train)
    val_set = set(groups_val)
    test_set = set(groups_test)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    assert not overlap_tv, f"Patient leakage: train ∩ val — {len(overlap_tv)} patients"
    assert not overlap_tt, f"Patient leakage: train ∩ test — {len(overlap_tt)} patients"
    assert not overlap_vt, f"Patient leakage: val ∩ test — {len(overlap_vt)} patients"

    logger.info(
        "Leakage check passed — %d train / %d val / %d test unique patients, all disjoint",
        len(train_set),
        len(val_set),
        len(test_set),
    )


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.load import load_raw
    from src.data.make_target import build_target
    from src.features.build_features import build_features

    df = build_target(load_raw())
    X, y, groups = build_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = patient_grouped_split(X, y, groups)

    print("\n--- Split shapes ---")
    for name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        print(f"  {name:5s}  X: {Xs.shape}  y: {ys.shape}  pos rate: {ys.mean():.4f}")

    # Recover per-split group arrays from the original groups ndarray via index alignment.
    # X_train.index etc. are integer positional indices into the original X.
    groups_train = groups[X_train.index]
    groups_val = groups[X_val.index]
    groups_test = groups[X_test.index]

    assert_no_patient_leakage(groups_train, groups_val, groups_test)
    print("\nLeakage assertion passed.")
