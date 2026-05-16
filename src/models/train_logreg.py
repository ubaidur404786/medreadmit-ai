"""Train the Logistic Regression baseline for 30-day readmission prediction (Module 1)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import assert_no_patient_leakage, patient_grouped_split
from src.evaluate.metrics import calibration_plot, evaluate_binary
from src.features.build_features import build_features

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# StandardScaler is mandatory before LogisticRegression — one-hot columns are
# already in {0, 1}, but numeric columns (time_in_hospital, num_medications,
# etc.) have different magnitudes that would bias the lbfgs optimiser.
PARAMS: dict[str, object] = dict(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    solver="lbfgs",
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

    assert_no_patient_leakage(
        groups[X_train.index],
        groups[X_val.index],
        groups[X_test.index],
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    mlflow.set_experiment("medreadmit-module1")

    with mlflow.start_run(run_name="logreg_baseline"):
        logger.info("Fitting Logistic Regression (StandardScaler → lbfgs, max_iter=2000)…")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**PARAMS)),
        ])
        model.fit(X_train, y_train)

        logger.info("Training complete.")

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
        mlflow.log_params({f"logreg_{k}": v for k, v in PARAMS.items()})
        mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})

        cal_path = MODELS_DIR / "logreg_val_calibration.png"
        calibration_plot(y_val.values, val_proba, n_bins=10, save_path=cal_path)
        mlflow.log_artifact(str(cal_path))

        model_path = MODELS_DIR / "logreg_baseline.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("Model artifact saved → %s", model_path)

        # ------------------------------------------------------------------
        # Summary table
        # ------------------------------------------------------------------
        _print_summary(train_metrics, val_metrics, test_metrics)


def _print_summary(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    """Print a formatted split-level metrics table to stdout."""
    header = f"{'Split':<6}  {'AUROC':>7}  {'AUPRC':>7}  {'Best-F1':>8}  {'Threshold':>9}"
    divider = "-" * len(header)

    print(f"\n{'Logistic Regression baseline results':^{len(header)}}")
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
    print()


if __name__ == "__main__":
    main()
