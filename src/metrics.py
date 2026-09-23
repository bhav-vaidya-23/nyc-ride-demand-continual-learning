"""
Evaluation metrics and subgroup benchmarking for demand forecasting.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute core forecasting metrics: MAE, RMSE, Mean Bias, and WAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ensure non-negative predictions for demand
    y_pred_clipped = np.maximum(0.0, y_pred)

    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    bias = float(np.mean(y_pred_clipped - y_true))

    total_demand = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred_clipped)) / total_demand) if total_demand > 0 else 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Bias": round(bias, 4),
        "WAPE": round(wape, 4),
    }


def evaluate_subgroups(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    high_demand_zones: List[int],
    peak_hours: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model metrics across high-demand vs low-demand zones and peak vs off-peak hours.
    """
    if peak_hours is None:
        peak_hours = [7, 8, 9, 17, 18, 19, 20]

    y_true = df["demand"].to_numpy(dtype=float)
    is_high = df["PULocationID"].isin(high_demand_zones).to_numpy()
    is_peak = df["hour"].isin(peak_hours).to_numpy()

    return {
        "Overall": calculate_metrics(y_true, y_pred),
        "High-Demand Zones": calculate_metrics(y_true[is_high], y_pred[is_high]),
        "Low-Demand Zones": calculate_metrics(y_true[~is_high], y_pred[~is_high]),
        "Peak Hours": calculate_metrics(y_true[is_peak], y_pred[is_peak]),
        "Off-Peak Hours": calculate_metrics(y_true[~is_peak], y_pred[~is_peak]),
    }


def build_comparison_table(
    models_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    baseline_key: str = "Naive (t-1)",
) -> pd.DataFrame:
    """
    Construct a structured comparison table showing MAE, RMSE, Bias, and % improvement.
    """
    results = []
    baseline_mae = None

    # First pass: find baseline MAE
    if baseline_key in models_dict:
        base_metrics = calculate_metrics(y_true, models_dict[baseline_key])
        baseline_mae = base_metrics["MAE"]

    for model_name, preds in models_dict.items():
        m = calculate_metrics(y_true, preds)
        row = {
            "Model": model_name,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "Bias": m["Bias"],
            "WAPE": m["WAPE"],
        }
        if baseline_mae is not None:
            pct_imprv = ((baseline_mae - m["MAE"]) / baseline_mae) * 100
            row["% MAE Improvement vs Baseline"] = round(pct_imprv, 2)
        else:
            row["% MAE Improvement vs Baseline"] = 0.0

        results.append(row)

    comp_df = pd.DataFrame(results)
    return comp_df
