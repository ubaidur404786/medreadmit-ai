"""Public PlattWrapper class for joblib-serialisable Platt-scaled classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattWrapper:
    """Wraps a base classifier with a fitted Platt scaling (sigmoid) calibration.

    The base model and a one-feature logistic regression are stored together so
    the artifact is self-contained: loading ``lgbm_calibrated.joblib`` gives a
    single object with a ``predict_proba`` interface, with no need to load the
    base model separately.

    This class lives in its own module so the pickle path
    (``src.models.platt_wrapper.PlattWrapper``) is stable — if the class were
    defined inside a script, renaming or moving that script would break
    deserialization of saved artifacts.

    Args:
        base_model: Any fitted classifier exposing ``predict_proba``.
        sigmoid: ``LogisticRegression`` fitted on the base model's raw scores
            (a single feature: the probability of the positive class).
    """

    def __init__(self, base_model: object, sigmoid: LogisticRegression) -> None:
        self.base_model = base_model
        self.sigmoid = sigmoid

    def predict_proba(self, X: object) -> np.ndarray:
        """Return calibrated class probabilities as an ``(n, 2)`` array.

        Args:
            X: Feature matrix accepted by ``base_model.predict_proba``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 is the calibrated
            probability of the positive (readmitted) class.
        """
        scores = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.sigmoid.predict_proba(scores)
