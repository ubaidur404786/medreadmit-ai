"""Train the LightGBM baseline for 30-day readmission prediction (Module 1)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np

from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import assert_no_patient_leakage, patient_grouped_split
from src.evaluate.metrics import calibration_plot, evaluate_binary
from src.features.build_features import build_features

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Baseline hyperparameters — tuned with Optuna on Day 5.
# class_weight="balanced" compensates for the ~11 % positive rate without
# requiring manual scale_pos_weight arithmetic.
PARAMS: dict[str, object] = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


def main() -> None:
    """Run the full train → evaluate → log pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    logger.info("Building feature matrix…")
    df = build_target(load_raw())
    X, y, groups = build_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = patient_grouped_split(X, y, groups)

    # Guard against patient leakage before spending time on training.
    assert_no_patient_leakage(
        groups[X_train.index],
        groups[X_val.index],
        groups[X_test.index],
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    mlflow.set_experiment("medreadmit-module1")

    with mlflow.start_run(run_name="lgbm_baseline"):
        logger.info(
            "Fitting LightGBM (n_estimators=%d, early stopping=50)…", PARAMS["n_estimators"]
        )

        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                # In LightGBM 4.x early stopping is a callback, not a constructor arg.
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100),
            ],
        )

        logger.info("Training complete — best iteration: %d", model.best_iteration_)

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        train_proba: np.ndarray = model.predict_proba(X_train)[:, 1]
        val_proba: np.ndarray = model.predict_proba(X_val)[:, 1]
        test_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

        train_metrics = evaluate_binary(y_train.values, train_proba, prefix="train")
        val_metrics = evaluate_binary(y_val.values, val_proba, prefix="val")
        test_metrics = evaluate_binary(y_test.values, test_proba, prefix="test")

        # ------------------------------------------------------------------
        # MLflow logging
        # ------------------------------------------------------------------
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})
        mlflow.log_metric("best_iteration", model.best_iteration_)

        # Calibration plot — saved to disk first, then registered as artifact.
        cal_path = MODELS_DIR / "lgbm_val_calibration.png"
        calibration_plot(y_val.values, val_proba, n_bins=10, save_path=cal_path)
        mlflow.log_artifact(str(cal_path))

        # Serialised model
        model_path = MODELS_DIR / "lgbm_baseline.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("Model artifact saved → %s", model_path)

        # ------------------------------------------------------------------
        # Summary table
        # ------------------------------------------------------------------
        _print_summary(train_metrics, val_metrics, test_metrics, model.best_iteration_)


def _print_summary(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_iteration: int,
) -> None:
    """Print a formatted split-level metrics table to stdout."""
    header = f"{'Split':<6}  {'AUROC':>7}  {'AUPRC':>7}  {'Best-F1':>8}  {'Threshold':>9}"
    divider = "-" * len(header)

    print(f"\n{'LightGBM baseline results':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(divider)

    for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        p = f"{split}_"
        print(
            f"{split:<6}  "
            f"{metrics[f'{p}auroc']:>7.4f}  "
            f"{metrics[f'{p}auprc']:>7.4f}  "
            f"{metrics[f'{p}best_f1']:>8.4f}  "
            f"{metrics[f'{p}best_threshold']:>9.4f}"
        )

    print("=" * len(header))
    print(f"  Best iteration (early stopping): {best_iteration}")
    print()


if __name__ == "__main__":
    main()
