"""
Model Training Pipeline
=======================
Trains XGBoost model with Optuna hyperparameter optimization using cross-validation.

Input: data/gold/
Output: results/
    - model.joblib (trained XGBoost model)
    - best_params.json (optimized hyperparameters)
    - metrics.json (cross-validation performance metrics)
    - feature_importance.csv (feature importance ranking)
    - cv_predictions.csv (cross-validation predictions)
    - cv_fold_metrics.csv (per-fold metrics)
    - predictions_ssp1_2_6.csv (scenario 1 predictions)
    - predictions_ssp2_4_5.csv (scenario 2 predictions)
    - predictions_ssp5_8_5.csv (scenario 3 predictions)
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

# Suppress Optuna logs during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_PATH = PROJECT_ROOT / "data" / "gold"
RESULTS_PATH = PROJECT_ROOT / "results"

# Feature columns to exclude from model input
NON_FEATURE_COLS = ["year", "yield"]

# Random seed for reproducibility
RANDOM_STATE = 42


def load_training_data() -> pd.DataFrame:
    """Load training data from gold layer."""
    training_path = GOLD_PATH / "training_data.parquet"

    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_path}")

    df = pd.read_parquet(training_path)
    print(f"  [OK] Loaded training data: {df.shape}")
    return df


def load_scenario_data(scenario: str) -> pd.DataFrame:
    """Load scenario data for predictions."""
    scenario_path = GOLD_PATH / f"{scenario}_features.parquet"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario data not found at {scenario_path}")

    df = pd.read_parquet(scenario_path)
    print(f"  [OK] Loaded {scenario}: {df.shape}")
    return df


def create_department_encoder(df: pd.DataFrame) -> dict:
    """Create a label encoder mapping for departments."""
    departments = sorted(df["nom_dep"].unique())
    return {dept: i for i, dept in enumerate(departments)}


def encode_department(df: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    """Encode nom_dep column using the provided encoder."""
    df = df.copy()
    df["nom_dep_encoded"] = df["nom_dep"].map(encoder)
    return df


def prepare_features(
    df: pd.DataFrame, dept_encoder: dict = None
) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """
    Prepare feature matrix X and target vector y.
    Returns (X, y, dept_encoder) for training data or (X, None, dept_encoder) for prediction data.
    """
    df = df.copy()

    # Create or use department encoder
    if dept_encoder is None:
        dept_encoder = create_department_encoder(df)

    # Encode department
    df = encode_department(df, dept_encoder)

    # Select feature columns (includes nom_dep_encoded)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c != "nom_dep"]

    X = df[feature_cols].copy()

    # Handle any remaining NaN values (from lag features in first year)
    X = X.fillna(X.median())

    if "yield" in df.columns:
        y = df["yield"].copy()
        return X, y, dept_encoder
    else:
        return X, None, dept_encoder


def create_optuna_objective(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5):
    """Create Optuna objective for XGBoost with anti-overfitting constraints."""

    def objective(trial):
        params = {
            # Reduced complexity
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # More aggressive subsampling (dropout-like)
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
            # Stronger leaf constraints
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            # Higher regularization (prevent overfitting)
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.1, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)

        # Cross-validation with negative MSE (sklearn convention)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="neg_mean_squared_error"
        )

        # Return mean RMSE (lower is better)
        rmse = np.sqrt(-scores.mean())
        return rmse

    return objective


def optimize_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 100, cv_folds: int = 5
) -> tuple[dict, optuna.Study]:
    """Run Optuna hyperparameter optimization."""
    print(f"  -> Running {n_trials} trials with {cv_folds}-fold CV...")

    study = optuna.create_study(direction="minimize", study_name="xgboost_yield")

    objective = create_optuna_objective(X_train, y_train, cv_folds)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  [OK] Best RMSE: {study.best_value:.4f}")
    print(f"  [OK] Best params: {study.best_params}")

    return study.best_params, study


def train_final_model(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: dict
) -> xgb.XGBRegressor:
    """Train final model with best hyperparameters."""
    params = {**best_params, "random_state": RANDOM_STATE, "n_jobs": -1}

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    return model


def get_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from model."""
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return importance


def generate_predictions(
    model: xgb.XGBRegressor, df: pd.DataFrame, feature_cols: list, dept_encoder: dict
) -> pd.DataFrame:
    """Generate predictions for a dataset."""
    X, _, _ = prepare_features(df, dept_encoder)

    # Ensure columns match training data
    X = X[feature_cols]

    predictions = model.predict(X)

    result = df[["nom_dep", "year"]].copy()
    result["yield_predicted"] = predictions

    return result


def save_results(
    model: xgb.XGBRegressor,
    best_params: dict,
    metrics: dict,
    feature_importance: pd.DataFrame,
    cv_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    scenario_predictions: dict[str, pd.DataFrame],
    dept_encoder: dict,
) -> None:
    """Save all results to results folder."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = RESULTS_PATH / "model.joblib"
    joblib.dump(model, model_path)
    print("  [OK] Saved model.joblib")

    # Save department encoder
    encoder_path = RESULTS_PATH / "dept_encoder.json"
    with open(encoder_path, "w") as f:
        json.dump(dept_encoder, f, indent=2)
    print("  [OK] Saved dept_encoder.json")

    # Save best params
    params_path = RESULTS_PATH / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print("  [OK] Saved best_params.json")

    # Save metrics
    metrics_path = RESULTS_PATH / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("  [OK] Saved metrics.json")

    # Save feature importance
    importance_path = RESULTS_PATH / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print("  [OK] Saved feature_importance.csv")

    # Save CV predictions
    cv_pred_path = RESULTS_PATH / "cv_predictions.csv"
    cv_predictions.to_csv(cv_pred_path, index=False)
    print("  [OK] Saved cv_predictions.csv")

    # Save fold metrics
    fold_path = RESULTS_PATH / "cv_fold_metrics.csv"
    fold_metrics.to_csv(fold_path, index=False)
    print("  [OK] Saved cv_fold_metrics.csv")

    # Save scenario predictions
    for scenario, preds in scenario_predictions.items():
        pred_path = RESULTS_PATH / f"predictions_{scenario}.csv"
        preds.to_csv(pred_path, index=False)
        print(f"  [OK] Saved predictions_{scenario}.csv")


def run_training(n_trials: int = 100, cv_folds: int = 5) -> dict:
    """
    Main training pipeline using K-fold cross-validation.

    Args:
        n_trials: Number of Optuna optimization trials
        cv_folds: Number of cross-validation folds
    """
    print("=" * 60)
    print("MODEL TRAINING PIPELINE (Cross-Validation)")
    print("=" * 60)

    # Step 1: Load training data
    print("\n[1/6] Loading training data...")
    df = load_training_data()

    # Step 2: Prepare features
    print("\n[2/6] Preparing features...")
    dept_encoder = create_department_encoder(df)
    X, y, _ = prepare_features(df, dept_encoder)
    feature_cols = X.columns.tolist()
    print(f"  -> Samples: {len(X)}, Features: {len(feature_cols)}")

    # Step 3: Optimize hyperparameters
    print(f"\n[3/6] Optimizing hyperparameters with {n_trials} Optuna trials...")
    best_params, study = optimize_hyperparameters(X, y, n_trials=n_trials, cv_folds=cv_folds)

    # Step 4: Run cross-validation with best hyperparameters
    print(f"\n[4/6] Running {cv_folds}-fold cross-validation with best params...")

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model = train_final_model(X_train, y_train, best_params)

        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        fold_metrics.append(
            {
                "fold": fold + 1,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "test_mae": test_mae,
            }
        )

        print(
            f"  Fold {fold + 1}: Train R2={train_r2:.4f}, "
            f"Test R2={test_r2:.4f}, RMSE={test_rmse:.4f}"
        )

        # Store predictions for analysis
        fold_preds = pd.DataFrame(
            {
                "nom_dep": df.iloc[test_idx]["nom_dep"].values,
                "year": df.iloc[test_idx]["year"].values,
                "yield_actual": y_test.values,
                "yield_predicted": y_test_pred,
                "fold": fold + 1,
            }
        )
        cv_results.append(fold_preds)

    # Aggregate metrics
    fold_df = pd.DataFrame(fold_metrics)
    cv_predictions = pd.concat(cv_results, ignore_index=True)

    metrics = {
        "cv": {
            "train_rmse_mean": float(fold_df["train_rmse"].mean()),
            "train_rmse_std": float(fold_df["train_rmse"].std()),
            "train_r2_mean": float(fold_df["train_r2"].mean()),
            "train_r2_std": float(fold_df["train_r2"].std()),
            "test_rmse_mean": float(fold_df["test_rmse"].mean()),
            "test_rmse_std": float(fold_df["test_rmse"].std()),
            "test_r2_mean": float(fold_df["test_r2"].mean()),
            "test_r2_std": float(fold_df["test_r2"].std()),
            "test_mae_mean": float(fold_df["test_mae"].mean()),
            "test_mae_std": float(fold_df["test_mae"].std()),
        }
    }

    # Step 5: Train final model on all data
    print("\n[5/6] Training final model on all data...")
    final_model = train_final_model(X, y, best_params)
    feature_importance = get_feature_importance(final_model, feature_cols)
    print("  [OK] Model trained on full dataset")

    # Print top features
    print("  -> Top 5 features:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    # Generate scenario predictions
    print("\n[6/6] Generating scenario predictions...")
    scenario_predictions = {}

    for scenario in ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]:
        try:
            scenario_df = load_scenario_data(scenario)
            preds = generate_predictions(final_model, scenario_df, feature_cols, dept_encoder)
            scenario_predictions[scenario] = preds
        except FileNotFoundError as e:
            print(f"  [WARN] Skipping {scenario}: {e}")

    # Save results
    print("\nSaving results...")
    save_results(
        model=final_model,
        best_params=best_params,
        metrics=metrics,
        feature_importance=feature_importance,
        cv_predictions=cv_predictions,
        fold_metrics=fold_df,
        scenario_predictions=scenario_predictions,
        dept_encoder=dept_encoder,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Summary
    cv = metrics["cv"]
    print(f"\n{cv_folds}-Fold Cross-Validation Results:")
    print(f"  Train RMSE: {cv['train_rmse_mean']:.4f} +/- {cv['train_rmse_std']:.4f}")
    print(f"  Train R2:   {cv['train_r2_mean']:.4f} +/- {cv['train_r2_std']:.4f}")
    print(f"  Test RMSE:  {cv['test_rmse_mean']:.4f} +/- {cv['test_rmse_std']:.4f}")
    print(f"  Test R2:    {cv['test_r2_mean']:.4f} +/- {cv['test_r2_std']:.4f}")
    print(f"  Test MAE:   {cv['test_mae_mean']:.4f} +/- {cv['test_mae_std']:.4f}")

    print(f"\nResults saved to: {RESULTS_PATH}")

    return {
        "model": final_model,
        "best_params": best_params,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "cv_predictions": cv_predictions,
        "fold_metrics": fold_df,
        "scenario_predictions": scenario_predictions,
    }


if __name__ == "__main__":
    run_training(n_trials=100, cv_folds=5)
