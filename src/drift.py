"""
Data drift and performance degradation monitoring utilities.
Explicitly distinguishes Data Drift P(X) from Concept Drift P(Y|X).
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between a reference (expected)
    and target (actual) feature distribution.

    Interpretation:
        PSI < 0.10: No significant shift (stable)
        0.10 <= PSI < 0.25: Moderate shift (monitor closely)
        PSI >= 0.25: Significant data drift detected
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bin edges from expected (reference) distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, percentiles)
    # Ensure strictly increasing bins
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate counts in each bucket
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division
    expected_pct = np.clip(expected_pct, epsilon, 1.0)
    actual_pct = np.clip(actual_pct, epsilon, 1.0)

    # Re-normalize
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    # Calculate PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))


def calculate_ks_test(
    reference: np.ndarray, current: np.ndarray
) -> Dict[str, float]:
    """
    Two-sample Kolmogorov-Smirnov test for continuous feature distribution shift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    stat, p_val = stats.ks_2samp(reference[~np.isnan(reference)], current[~np.isnan(current)])
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 6),
        "is_drift": bool(p_val < 0.05 and stat > 0.1),
    }


def compute_rolling_metrics(
    predictions_df: pd.DataFrame, window_size: int = 96
) -> pd.DataFrame:
    """
    Compute rolling performance metrics (MAE, RMSE, Bias) over sliding windows of predictions.
    Default window_size=96 corresponds to 24 hours of 15-min intervals per zone.
    """
    df = predictions_df.copy().sort_values("prediction_timestamp").reset_index(drop=True)
    df["error"] = df["predicted_demand"] - df["actual_demand"]
    df["abs_error"] = np.abs(df["error"])
    df["sq_error"] = df["error"] ** 2

    # Group by timestamp for system-wide rolling metrics
    ts_summary = df.groupby("prediction_timestamp").agg(
        mean_abs_error=("abs_error", "mean"),
        mean_sq_error=("sq_error", "mean"),
        mean_bias=("error", "mean"),
        total_actual=("actual_demand", "sum"),
    ).reset_index()

    ts_summary["rolling_mae"] = ts_summary["mean_abs_error"].rolling(window_size, min_periods=4).mean()
    ts_summary["rolling_rmse"] = np.sqrt(ts_summary["mean_sq_error"].rolling(window_size, min_periods=4).mean())
    ts_summary["rolling_bias"] = ts_summary["mean_bias"].rolling(window_size, min_periods=4).mean()

    return ts_summary


def detect_sustained_degradation(
    rolling_mae_series: pd.Series,
    baseline_mae: float,
    threshold_pct: float = 0.15,
    consecutive_periods: int = 4,
) -> Tuple[bool, List[int]]:
    """
    Detect whether rolling MAE sustained a degradation exceeding threshold_pct
    above baseline_mae for consecutive_periods.
    """
    limit = baseline_mae * (1.0 + threshold_pct)
    exceeded = (rolling_mae_series > limit).to_numpy()

    degraded_indices = []
    consecutive = 0
    for idx, is_exceeded in enumerate(exceeded):
        if is_exceeded:
            consecutive += 1
            if consecutive >= consecutive_periods:
                degraded_indices.append(idx)
        else:
            consecutive = 0

    has_sustained_degradation = len(degraded_indices) > 0
    return has_sustained_degradation, degraded_indices
