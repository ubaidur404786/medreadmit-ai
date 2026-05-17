"""Post-hoc Platt scaling calibration for the LightGBM baseline model (Module 1).

``class_weight="balanced"`` shifts predicted probabilities far above the true
base rate (~0.11).  This script fits a sigmoid (Platt scaling) on the held-out
validation set and saves the calibrated model for downstream use.

Why the *baseline* model and not the tuned one:
    The tuned model was retrained on train+val combined, so no clean held-out
    set is available to fit the sigmoid without data leakage.  The baseline was
    trained on train only, making val genuinely unseen and safe for calibration.
    The calibrated baseline is the conservative, production-safe artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.load import load_raw
from src.models.platt_wrapper import PlattWrapper
from src.data.make_target import build_target
from src.data.split import assert_no_patient_leakage, patient_grouped_split
from src.evaluate.metrics import calibration_plot, evaluate_binary
from src.features.build_features import build_features

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_BASELINE_PATH = MODELS_DIR / "lgbm_baseline.joblib"
_CALIBRATED_PATH = MODELS_DIR / "lgbm_calibrated.joblib"


def main() -> None:
    """Fit Platt scaling on val set and evaluate before/after on test."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not _BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {_BASELINE_PATH}. "
            "Run `python -m src.models.train_lgbm` first."
        )

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    logger.info("Building feature matrix…")
    df = build_target(load_raw())
    X, y, groups = build_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = patient_grouped_split(X, y, groups)
    assert_no_patient_leakage(
        groups[X_train.index],
        groups[X_val.index],
        groups[X_test.index],
    )

    # ------------------------------------------------------------------
    # Load baseline and record uncalibrated test predictions
    # ------------------------------------------------------------------
    logger.info("Loading baseline model from %s", _BASELINE_PATH)
    baseline_model = joblib.load(_BASELINE_PATH)

    raw_proba: np.ndarray = baseline_model.predict_proba(X_test)[:, 1]
    logger.info(
        "Baseline raw — mean predicted: %.4f  true base rate: %.4f",
        raw_proba.mean(),
        y_test.mean(),
    )

    # ------------------------------------------------------------------
    # Platt scaling: fit sigmoid on val (unseen by baseline model)
    # CalibratedClassifierCV(cv="prefit") was removed in sklearn 1.5, so we
    # replicate what it did internally: fit a LogisticRegression on the raw
    # scores, which is exactly Platt's original formulation.
    # ------------------------------------------------------------------
    logger.info("Fitting Platt scaling on val set (%d rows)…", len(X_val))
    val_scores = baseline_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    sigmoid = LogisticRegression(C=1e10, solver="lbfgs", random_state=42)
    sigmoid.fit(val_scores, y_val)

    calibrated_model = PlattWrapper(baseline_model, sigmoid)

    cal_proba: np.ndarray = calibrated_model.predict_proba(X_test)[:, 1]
    logger.info(
        "Calibrated    — mean predicted: %.4f  true base rate: %.4f",
        cal_proba.mean(),
        y_test.mean(),
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    raw_metrics = evaluate_binary(y_test.values, raw_proba, prefix="test")
    cal_metrics = evaluate_binary(y_test.values, cal_proba, prefix="test")

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    mlflow.set_experiment("medreadmit-module1")

    with mlflow.start_run(run_name="lgbm_calibrated"):
        mlflow.log_params(
            {
                "base_model": "lgbm_baseline",
                "calibration_method": "sigmoid",
                "calibration_cv": "manual_platt",
                "calibration_set": "val",
            }
        )

        mlflow.log_metrics(
            {
                "raw_test_auroc": raw_metrics["test_auroc"],
                "raw_test_auprc": raw_metrics["test_auprc"],
                "raw_test_brier": raw_metrics["test_brier"],
                "raw_mean_predicted": float(raw_proba.mean()),
                "cal_test_auroc": cal_metrics["test_auroc"],
                "cal_test_auprc": cal_metrics["test_auprc"],
                "cal_test_brier": cal_metrics["test_brier"],
                "cal_mean_predicted": float(cal_proba.mean()),
                "true_base_rate": float(y_test.mean()),
                "brier_reduction": raw_metrics["test_brier"] - cal_metrics["test_brier"],
            }
        )

        cal_plot_path = MODELS_DIR / "lgbm_calibrated_test_calibration.png"
        calibration_plot(y_test.values, cal_proba, n_bins=10, save_path=cal_plot_path)
        mlflow.log_artifact(str(cal_plot_path))

        joblib.dump(calibrated_model, _CALIBRATED_PATH)
        mlflow.log_artifact(str(_CALIBRATED_PATH))
        logger.info("Calibrated model saved → %s", _CALIBRATED_PATH)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    _print_summary(raw_metrics, cal_metrics, raw_proba, cal_proba, float(y_test.mean()))


def _print_summary(
    raw_metrics: dict[str, float],
    cal_metrics: dict[str, float],
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    base_rate: float,
) -> None:
    """Print a before/after calibration comparison table."""
    header = (
        f"{'Model':<24}  {'AUROC':>7}  {'AUPRC':>7}  {'Brier':>7}"
        f"  {'Mean pred':>9}  {'Mean obs':>8}"
    )
    divider = "-" * len(header)

    print(f"\n{'Calibration results (test set)':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(divider)

    rows = [
        ("lgbm_baseline (raw)", raw_metrics, float(raw_proba.mean())),
        ("lgbm_calibrated", cal_metrics, float(cal_proba.mean())),
    ]
    for label, m, mean_pred in rows:
        print(
            f"{label:<24}  "
            f"{m['test_auroc']:>7.4f}  "
            f"{m['test_auprc']:>7.4f}  "
            f"{m['test_brier']:>7.4f}  "
            f"{mean_pred:>9.4f}  "
            f"{base_rate:>8.4f}"
        )

    print("=" * len(header))
    brier_delta = raw_metrics["test_brier"] - cal_metrics["test_brier"]
    print(f"  Brier improvement: {brier_delta:+.4f}  (negative = calibration got worse)")
    print()


if __name__ == "__main__":
    main()
