"""Generate SHAP explanations for the tuned LightGBM model on the test set.

Produces four plots and a feature-importance CSV, all saved to reports/ and
logged to MLflow.  Run from the repo root after tune_lgbm.py has completed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import shap

from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import patient_grouped_split
from src.explain.shap_utils import (
    compute_shap_values,
    load_tuned_model,
    plot_importance_bar,
    plot_summary,
    plot_waterfall,
)
from src.features.build_features import build_features

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/lgbm_tuned.joblib")
_FIGURES_DIR = Path("reports/figures/shap")
_REPORTS_DIR = Path("reports")


def _shap_for_patient(
    explainer: shap.TreeExplainer,
    X_row: "pd.DataFrame",  # noqa: F821  (pd imported inside main to keep top-level light)
) -> np.ndarray:
    """Return a 2-D SHAP array of shape (1, n_features) for a single patient.

    Mirrors the normalisation in :func:`~src.explain.shap_utils.compute_shap_values`
    so waterfall plots are consistent with the global explanations.

    Args:
        explainer: Fitted :class:`~shap.TreeExplainer`.
        X_row: Single-row DataFrame for the patient to explain.

    Returns:
        Float array of shape ``(1, n_features)`` for the positive class.
    """
    raw = explainer.shap_values(X_row)
    if isinstance(raw, list):
        sv = raw[1]
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        sv = raw[:, :, 1]
    else:
        sv = raw
    return sv  # shape (1, n_features)


def main() -> None:
    """Run SHAP analysis on the test set and log artifacts to MLflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not _MODEL_PATH.exists():
        logger.error(
            "Tuned model not found at %s. Run `python -m src.models.tune_lgbm` first.",
            _MODEL_PATH,
        )
        sys.exit(1)

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data pipeline — identical to the training scripts
    # ------------------------------------------------------------------
    logger.info("Building feature matrix…")
    df = build_target(load_raw())
    X, y, groups = build_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = patient_grouped_split(X, y, groups)

    # ------------------------------------------------------------------
    # Model + probabilities
    # ------------------------------------------------------------------
    logger.info("Loading tuned model from %s", _MODEL_PATH)
    model = load_tuned_model(_MODEL_PATH)

    test_proba: np.ndarray = model.predict_proba(X_test)[:, 1]
    logger.info(
        "Test set: %d rows — mean predicted risk %.4f, max %.4f",
        len(X_test),
        test_proba.mean(),
        test_proba.max(),
    )

    # ------------------------------------------------------------------
    # SHAP values on a 5000-row test sample (global explanations)
    # Explaining on held-out data reflects generalisation, not memorisation.
    # ------------------------------------------------------------------
    logger.info("Computing SHAP values on 5000-row test sample…")
    shap_values, X_sample, explainer = compute_shap_values(
        model, X_test, sample_size=5000, random_state=42
    )

    # ------------------------------------------------------------------
    # Global plots
    # ------------------------------------------------------------------
    logger.info("Saving beeswarm summary plot…")
    plot_summary(
        shap_values, X_sample,
        max_display=20,
        save_path=_FIGURES_DIR / "summary_beeswarm.png",
    )

    logger.info("Saving feature importance bar plot…")
    importance_df = plot_importance_bar(
        shap_values, X_sample,
        max_display=20,
        save_path=_FIGURES_DIR / "importance_bar.png",
    )

    # ------------------------------------------------------------------
    # Waterfall — highest-risk patient in test set
    # ------------------------------------------------------------------
    high_risk_pos = int(np.argmax(test_proba))
    high_risk_prob = float(test_proba[high_risk_pos])
    high_risk_label = int(y_test.values[high_risk_pos])
    logger.info(
        "High-risk patient: iloc=%d  pred=%.4f  true_label=%d",
        high_risk_pos, high_risk_prob, high_risk_label,
    )

    high_risk_row = X_test.iloc[[high_risk_pos]]
    high_risk_shap = _shap_for_patient(explainer, high_risk_row)
    plot_waterfall(
        explainer, high_risk_shap, high_risk_row,
        patient_idx=0,
        save_path=_FIGURES_DIR / "waterfall_high_risk.png",
    )

    # ------------------------------------------------------------------
    # Waterfall — low-risk patient: pred ≈ 0.05 AND actually not readmitted.
    # "Model agrees with reality" makes the explanation a clean sanity check.
    # ------------------------------------------------------------------
    negative_positions = np.where(y_test.values == 0)[0]
    low_risk_pos = int(
        negative_positions[np.argmin(np.abs(test_proba[negative_positions] - 0.05))]
    )
    low_risk_prob = float(test_proba[low_risk_pos])
    logger.info(
        "Low-risk patient: iloc=%d  pred=%.4f  true_label=0",
        low_risk_pos, low_risk_prob,
    )

    low_risk_row = X_test.iloc[[low_risk_pos]]
    low_risk_shap = _shap_for_patient(explainer, low_risk_row)
    plot_waterfall(
        explainer, low_risk_shap, low_risk_row,
        patient_idx=0,
        save_path=_FIGURES_DIR / "waterfall_low_risk.png",
    )

    # ------------------------------------------------------------------
    # Feature importance CSV
    # ------------------------------------------------------------------
    csv_path = _REPORTS_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.info("Feature importance saved → %s", csv_path)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\nTop-10 features by mean |SHAP|:")
    print(
        importance_df.head(10)
        .assign(mean_abs_shap=lambda d: d["mean_abs_shap"].map("{:.5f}".format))
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    mlflow.set_experiment("medreadmit-module1")
    with mlflow.start_run(run_name="shap_analysis"):
        mlflow.log_metric("n_shap_samples", len(X_sample))
        mlflow.log_metric("high_risk_pred_prob", high_risk_prob)
        mlflow.log_metric("low_risk_pred_prob", low_risk_prob)

        for plot_path in [
            _FIGURES_DIR / "summary_beeswarm.png",
            _FIGURES_DIR / "importance_bar.png",
            _FIGURES_DIR / "waterfall_high_risk.png",
            _FIGURES_DIR / "waterfall_low_risk.png",
        ]:
            mlflow.log_artifact(str(plot_path), artifact_path="shap_plots")

        mlflow.log_artifact(str(csv_path))
        logger.info("All artifacts logged to MLflow run 'shap_analysis'.")


if __name__ == "__main__":
    main()
