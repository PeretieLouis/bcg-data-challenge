"""
Model Evaluation Module
=======================
Analysis utilities for evaluating trained models and exploring predictions.
Loads results from the train.py output and provides analysis functions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results"
SILVER_PATH = PROJECT_ROOT / "data" / "silver"


def load_metrics() -> dict:
    """Load metrics from results."""
    metrics_path = RESULTS_PATH / "metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


def load_feature_importance() -> pd.DataFrame:
    """Load feature importance from results."""
    importance_path = RESULTS_PATH / "feature_importance.csv"
    return pd.read_csv(importance_path)


def load_cv_predictions() -> pd.DataFrame:
    """Load cross-validation predictions from results."""
    pred_path = RESULTS_PATH / "cv_predictions.csv"
    return pd.read_csv(pred_path)


def load_scenario_predictions(scenario: str) -> pd.DataFrame:
    """Load scenario predictions from results."""
    pred_path = RESULTS_PATH / f"predictions_{scenario}.csv"
    return pd.read_csv(pred_path)


def load_fold_metrics() -> pd.DataFrame:
    """Load per-fold metrics from results."""
    fold_path = RESULTS_PATH / "cv_fold_metrics.csv"
    return pd.read_csv(fold_path)


def analyze_predictions_by_department(predictions: pd.DataFrame) -> pd.DataFrame:
    """Analyze prediction performance by department."""
    if "yield_actual" not in predictions.columns:
        raise ValueError("Predictions must include 'yield_actual' column")

    dept_metrics = []
    for dept in predictions["nom_dep"].unique():
        dept_data = predictions[predictions["nom_dep"] == dept]
        y_true = dept_data["yield_actual"]
        y_pred = dept_data["yield_predicted"]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        dept_metrics.append(
            {
                "nom_dep": dept,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_samples": len(dept_data),
            }
        )

    return pd.DataFrame(dept_metrics).sort_values("rmse")


def analyze_predictions_by_year(predictions: pd.DataFrame) -> pd.DataFrame:
    """Analyze prediction performance by year."""
    if "yield_actual" not in predictions.columns:
        raise ValueError("Predictions must include 'yield_actual' column")

    year_metrics = []
    for year in sorted(predictions["year"].unique()):
        year_data = predictions[predictions["year"] == year]
        y_true = year_data["yield_actual"]
        y_pred = year_data["yield_predicted"]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        year_metrics.append(
            {
                "year": year,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_samples": len(year_data),
            }
        )

    return pd.DataFrame(year_metrics)


def get_scenario_summary(scenario: str) -> pd.DataFrame:
    """Get summary statistics for scenario predictions."""
    preds = load_scenario_predictions(scenario)

    summary = (
        preds.groupby("year")
        .agg(
            mean_yield=("yield_predicted", "mean"),
            std_yield=("yield_predicted", "std"),
            min_yield=("yield_predicted", "min"),
            max_yield=("yield_predicted", "max"),
        )
        .reset_index()
    )

    return summary


def compare_scenarios() -> pd.DataFrame:
    """Compare yield predictions across all scenarios."""
    all_scenarios = []

    for scenario in ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]:
        try:
            preds = load_scenario_predictions(scenario)
            yearly_mean = preds.groupby("year")["yield_predicted"].mean().reset_index()
            yearly_mean["scenario"] = scenario
            all_scenarios.append(yearly_mean)
        except FileNotFoundError:
            continue

    if all_scenarios:
        return pd.concat(all_scenarios, ignore_index=True)
    return pd.DataFrame()


def add_area_context(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add area data from silver layer for business context."""
    yield_data = pd.read_parquet(SILVER_PATH / "barley_yield.parquet")

    # Get average area per department
    avg_area = yield_data.groupby("nom_dep")["area"].mean().reset_index()
    avg_area.columns = ["nom_dep", "avg_area_ha"]

    return predictions.merge(avg_area, on="nom_dep", how="left")


def print_evaluation_summary():
    """Print a summary of model evaluation."""
    print("=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)

    # Load metrics
    metrics = load_metrics()

    if "cv" in metrics:
        print("\nCross-Validation Performance:")
        cv = metrics["cv"]
        print(f"  Train RMSE: {cv['train_rmse_mean']:.4f} +/- {cv['train_rmse_std']:.4f}")
        print(f"  Train R2:   {cv['train_r2_mean']:.4f} +/- {cv['train_r2_std']:.4f}")
        print(f"  Test RMSE:  {cv['test_rmse_mean']:.4f} +/- {cv['test_rmse_std']:.4f}")
        print(f"  Test R2:    {cv['test_r2_mean']:.4f} +/- {cv['test_r2_std']:.4f}")
        print(f"  Test MAE:   {cv['test_mae_mean']:.4f} +/- {cv['test_mae_std']:.4f}")

    # Load feature importance
    importance = load_feature_importance()
    print("\nTop 10 Features:")
    for i, row in importance.head(10).iterrows():
        print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

    # CV predictions analysis
    try:
        cv_preds = load_cv_predictions()
        dept_analysis = analyze_predictions_by_department(cv_preds)

        print("\nBest Predicted Departments (lowest RMSE):")
        for _, row in dept_analysis.head(5).iterrows():
            print(f"  {row['nom_dep']}: RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}")

        print("\nWorst Predicted Departments (highest RMSE):")
        for _, row in dept_analysis.tail(5).iterrows():
            print(f"  {row['nom_dep']}: RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}")
    except FileNotFoundError:
        print("\n[WARN] CV predictions not found")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_evaluation_summary()
