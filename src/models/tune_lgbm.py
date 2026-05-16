"""Optuna hyperparameter search for LightGBM readmission model (Module 1)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna_integration.mlflow import MLflowCallback
from sklearn.metrics import roc_auc_score

from src.data.load import load_raw
from src.data.make_target import build_target
from src.data.split import assert_no_patient_leakage, patient_grouped_split
from src.evaluate.metrics import calibration_plot, evaluate_binary
from src.features.build_features import build_features

# Suppress per-trial INFO lines — the progress bar is sufficient feedback.
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
N_TRIALS = 50


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Evaluate a single LightGBM hyperparameter configuration on the val set.

    The best early-stopping iteration is stashed in a trial user attribute so
    the final retrain can replicate it without a held-out val set.

    Args:
        trial: Active Optuna trial used to sample hyperparameters.
        X_train: Training feature matrix.
        y_train: Training target series.
        X_val: Validation feature matrix.
        y_val: Validation target series.

    Returns:
        Validation AUROC (higher is better).
    """
    params = dict(
        n_estimators=2000,  # high ceiling; early stopping decides the actual count
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 128, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    trial.set_user_attr("best_iteration", model.best_iteration_)

    val_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_proba)


def main() -> None:
    """Run Optuna search, retrain best config on train+val, evaluate on test."""
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
    # Optuna study
    # ------------------------------------------------------------------
    mlflow.set_experiment("medreadmit-module1")

    # Each trial is logged as a nested child run of the parent "lgbm_optuna_best" run.
    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="val_auroc",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        study_name="lgbm_readmit",
    )

    logger.info("Starting Optuna search (%d trials)…", N_TRIALS)

    with mlflow.start_run(run_name="lgbm_optuna_best"):
        study.optimize(
            lambda t: objective(t, X_train, y_train, X_val, y_val),
            n_trials=N_TRIALS,
            show_progress_bar=True,
            callbacks=[mlflow_cb],
        )

        best_params: dict[str, object] = study.best_params
        best_val_auroc: float = study.best_value
        # Use the iteration count the best trial's model converged to — training
        # the final model to this count on train+val avoids over/underfitting
        # relative to the tuned configuration.
        best_n_estimators: int = study.best_trial.user_attrs["best_iteration"]

        logger.info("Best val AUROC: %.4f", best_val_auroc)
        logger.info("Best n_estimators (early stopping): %d", best_n_estimators)
        logger.info("Best params: %s", best_params)

        # ------------------------------------------------------------------
        # Final retrain on train + val
        # ------------------------------------------------------------------
        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])
        logger.info("Retraining final model on train+val (%d rows)…", len(X_trainval))

        final_params = {
            **best_params,
            "n_estimators": best_n_estimators,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(X_trainval, y_trainval)

        # ------------------------------------------------------------------
        # Evaluation on test
        # ------------------------------------------------------------------
        test_proba: np.ndarray = final_model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_binary(y_test.values, test_proba, prefix="test")

        # ------------------------------------------------------------------
        # MLflow logging
        # ------------------------------------------------------------------
        mlflow.log_metric("best_val_auroc", best_val_auroc)
        mlflow.log_metric("best_n_estimators", best_n_estimators)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(test_metrics)

        cal_path = MODELS_DIR / "lgbm_tuned_test_calibration.png"
        calibration_plot(y_test.values, test_proba, n_bins=10, save_path=cal_path)
        mlflow.log_artifact(str(cal_path))

        model_path = MODELS_DIR / "lgbm_tuned.joblib"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("Tuned model saved → %s", model_path)

        # ------------------------------------------------------------------
        # Comparison table
        # ------------------------------------------------------------------
        baseline_metrics = _fetch_baseline_metrics()
        _print_summary(best_val_auroc, test_metrics, baseline_metrics)


def _fetch_baseline_metrics() -> dict[str, float] | None:
    """Retrieve the most recent lgbm_baseline test metrics from MLflow.

    Returns:
        Dict with keys ``test_auroc``, ``test_auprc``, ``test_best_f1``, or
        ``None`` if the run cannot be found.
    """
    try:
        runs = mlflow.search_runs(
            experiment_names=["medreadmit-module1"],
            order_by=["start_time DESC"],
        )
        baseline_runs = runs[runs["tags.mlflow.runName"] == "lgbm_baseline"]
        if baseline_runs.empty:
            return None
        row = baseline_runs.iloc[0]
        return {
            "test_auroc": float(row["metrics.test_auroc"]),
            "test_auprc": float(row["metrics.test_auprc"]),
            "test_best_f1": float(row["metrics.test_best_f1"]),
        }
    except Exception as exc:
        logger.warning("Could not fetch baseline metrics from MLflow: %s", exc)
        return None


def _print_summary(
    best_val_auroc: float,
    test_metrics: dict[str, float],
    baseline: dict[str, float] | None,
) -> None:
    """Print a comparison table of baseline vs tuned LightGBM on the test set."""
    header = (
        f"{'Model':<20}  {'Val AUROC':>9}  {'Test AUROC':>10}  {'Test AUPRC':>10}  {'Test F1':>8}"
    )
    divider = "-" * len(header)

    print(f"\n{'LightGBM tuning results':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(divider)

    if baseline is not None:
        print(
            f"{'lgbm_baseline':<20}  {'—':>9}  "
            f"{baseline['test_auroc']:>10.4f}  "
            f"{baseline['test_auprc']:>10.4f}  "
            f"{baseline['test_best_f1']:>8.4f}"
        )

    print(
        f"{'lgbm_tuned':<20}  {best_val_auroc:>9.4f}  "
        f"{test_metrics['test_auroc']:>10.4f}  "
        f"{test_metrics['test_auprc']:>10.4f}  "
        f"{test_metrics['test_best_f1']:>8.4f}"
    )
    print("=" * len(header))
    print()


if __name__ == "__main__":
    main()
