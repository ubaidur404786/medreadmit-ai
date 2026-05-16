"""SHAP explainability utilities for the tuned LightGBM readmission model."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Applied at import time so every SHAP plot inherits a reasonable canvas size.
# Downstream callers (Streamlit, scripts) can override per-figure if needed.
plt.rcParams["figure.figsize"] = (10, 8)


def load_tuned_model(path: Path = Path("models/lgbm_tuned.joblib")) -> lgb.LGBMClassifier:
    """Load the tuned LightGBM classifier saved by ``tune_lgbm.py``.

    Args:
        path: Path to the ``.joblib`` file. Resolved relative to the caller's
            working directory; pass an absolute path when calling from outside
            the repo root.

    Returns:
        Fitted :class:`lightgbm.LGBMClassifier` instance.
    """
    return joblib.load(path)


def compute_shap_values(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    sample_size: int | None = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, shap.TreeExplainer]:
    """Compute TreeExplainer SHAP values for the positive readmission class.

    For global explanations (summary / importance plots) a random subsample of
    ``sample_size`` rows is used to keep runtime and memory reasonable on an
    8 GB machine.  For per-patient waterfall plots pass ``sample_size=None`` or
    call this with the full X and index directly in :func:`plot_waterfall`.

    The function handles the API variance across SHAP versions:

    * 3-D array ``(n, f, 2)`` — slices ``[:, :, 1]`` for the positive class.
    * List of length 2 — takes element ``[1]``.
    * 2-D array ``(n, f)`` — used directly (already the positive class).

    Args:
        model: Fitted :class:`~lightgbm.LGBMClassifier`.
        X: Feature matrix with column names preserved (required for plots).
        sample_size: Number of rows to subsample for global explanations.
            Pass ``None`` to use the full matrix.
        random_state: Seed for the subsample RNG.

    Returns:
        Tuple of ``(shap_values, X_sample, explainer)`` where ``shap_values``
        is a 2-D float array of shape ``(n_sample, n_features)``.
    """
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)

    # Normalise to a 2-D (n_samples, n_features) array for the positive class.
    if isinstance(raw, list):
        # Older SHAP: list of [class-0 array, class-1 array]
        shap_values = raw[1]
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        # Some SHAP builds: shape (n_samples, n_features, n_classes)
        shap_values = raw[:, :, 1]
    else:
        shap_values = raw  # already (n_samples, n_features) for binary

    assert shap_values.shape == (len(X_sample), X_sample.shape[1]), (
        f"Unexpected SHAP output shape {shap_values.shape}; "
        f"expected ({len(X_sample)}, {X_sample.shape[1]})"
    )

    return shap_values, X_sample, explainer


def plot_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 20,
    save_path: Path | None = None,
) -> None:
    """Beeswarm summary plot of SHAP values for the top features.

    Each dot is one patient coloured by feature value (red = high, blue = low).
    The horizontal axis shows SHAP impact on the log-odds of readmission.

    Args:
        shap_values: 2-D array ``(n_samples, n_features)`` from
            :func:`compute_shap_values`.
        X_sample: Feature matrix aligned with ``shap_values``.
        max_display: Number of top features to show.
        save_path: If provided, the figure is saved here (parent dirs created).
    """
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_importance_bar(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 20,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Bar chart of mean absolute SHAP values (global feature importance).

    Unlike the built-in LightGBM importance (split count or gain), mean |SHAP|
    measures the actual average *impact* of each feature on model predictions
    in the units of log-odds — more interpretable for clinicians.

    Args:
        shap_values: 2-D array ``(n_samples, n_features)`` from
            :func:`compute_shap_values`.
        X_sample: Feature matrix aligned with ``shap_values``.
        max_display: Number of top features to plot.
        save_path: If provided, the figure is saved here (parent dirs created).

    Returns:
        DataFrame with columns ``feature`` and ``mean_abs_shap``, sorted
        descending — the full ranking, not just the plotted top-N.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return importance


def plot_waterfall(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    patient_idx: int,
    save_path: Path | None = None,
) -> None:
    """SHAP waterfall plot explaining a single patient's readmission prediction.

    Uses the SHAP 0.45+ :class:`~shap.Explanation` API so that feature names
    and base values are embedded in the plot without requiring global state.

    Args:
        explainer: :class:`~shap.TreeExplainer` returned by
            :func:`compute_shap_values`.
        shap_values: 2-D array ``(n_samples, n_features)`` — the same array
            passed to other plot functions.  ``patient_idx`` indexes into this.
        X: Feature matrix aligned with ``shap_values`` (must have the same
            index ordering).
        patient_idx: Row index into ``shap_values`` / ``X`` to explain.
        save_path: If provided, the figure is saved here (parent dirs created).
    """
    # expected_value is a list [class-0, class-1] for binary classifiers.
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(base[1])

    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=base,
        data=X.iloc[patient_idx].values,
        feature_names=X.columns.tolist(),
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
