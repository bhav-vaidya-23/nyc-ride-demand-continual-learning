"""
Leak-free feature engineering for spatio-temporal ride demand forecasting.
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from src.config import LAG_STEPS, ROLLING_WINDOWS


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal calendar features from timestamp.
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["dayofweek"] = ts.dt.dayofweek.astype("int8")  # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["time_slot_of_day"] = (df["hour"] * 4 + df["minute"] // 15).astype("int16")
    return df


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    lag_steps: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    group_col: str = "PULocationID",
    target_col: str = "demand",
) -> pd.DataFrame:
    """
    Generate lag and rolling features strictly using historical values (t-1 and older).
    Guarantees ZERO lookahead / data leakage.

    Args:
        df: DataFrame sorted by [group_col, 'timestamp'].
        lag_steps: Intervals to lag. Defaults to [1, 2, 4, 8, 96, 672].
        rolling_windows: Windows to roll over shifted series (e.g., [4, 8]).
        group_col: Zone grouping column.
        target_col: Target demand column.

    Returns:
        DataFrame with added lag and rolling columns.
    """
    if lag_steps is None:
        lag_steps = LAG_STEPS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    df = df.sort_values(by=[group_col, "timestamp"]).copy()

    grouped = df.groupby(group_col)[target_col]

    # Compute lags: shift(k) means value from k intervals ago
    for lag in lag_steps:
        col_name = f"lag_{lag}"
        df[col_name] = grouped.shift(lag)

    # Compute short-term demand trend: demand(t-1) - demand(t-2)
    if 1 in lag_steps and 2 in lag_steps:
        df["demand_trend_15m"] = df["lag_1"] - df["lag_2"]

    # Compute rolling mean and std over shifted series to ensure no leakage of t
    # shift(1) ensures the window only covers t-1, t-2, ...
    shifted_target = grouped.shift(1)
    for window in rolling_windows:
        mean_col = f"rolling_mean_{window}"
        std_col = f"rolling_std_{window}"

        # Using transform with rolling on grouped shifted series
        df[mean_col] = (
            shifted_target.groupby(df[group_col])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[std_col] = (
            shifted_target.groupby(df[group_col])
            .rolling(window, min_periods=1)
            .std()
            .fillna(0.0)
            .reset_index(level=0, drop=True)
        )

    return df


def build_feature_pipeline(
    df: pd.DataFrame, drop_burn_in: bool = True
) -> pd.DataFrame:
    """
    Full feature pipeline combining calendar, lag, and rolling features.

    Args:
        df: Clean 15-min demand grid.
        drop_burn_in: If True, drops initial records where the maximum lag is NaN.

    Returns:
        Feature-enriched DataFrame.
    """
    df = create_calendar_features(df)
    df = create_lag_and_rolling_features(df)

    if drop_burn_in:
        max_lag = max(LAG_STEPS)
        # Drop rows where lag_672 (or largest lag) is NaN
        df = df.dropna(subset=[f"lag_{max_lag}"]).reset_index(drop=True)

    return df


def get_feature_columns() -> List[str]:
    """Return the canonical list of ML feature column names."""
    features = [
        "PULocationID",
        "hour",
        "minute",
        "dayofweek",
        "is_weekend",
        "time_slot_of_day",
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "lag_96",
        "lag_672",
        "demand_trend_15m",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_8",
        "rolling_std_8",
    ]
    return features
