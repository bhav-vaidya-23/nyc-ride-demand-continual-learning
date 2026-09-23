"""
Baseline forecasting models for ride demand prediction.
Philosophical core: determine whether ML provides genuine improvement over these strong baselines.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class NaiveLastPeriodBaseline:
    """
    Baseline A: Naive previous-period forecast.
    demand_pred(t) = demand(t-1)
    """

    def fit(self, train_df: pd.DataFrame) -> "NaiveLastPeriodBaseline":
        # Parameter-free baseline
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "lag_1" in df.columns:
            return df["lag_1"].to_numpy(dtype=float)
        raise ValueError("DataFrame must contain 'lag_1' column to evaluate Naive baseline.")


class HistoricalSeasonalBaseline:
    """
    Baseline B: Historical / seasonal lookup baseline.
    Predicts the historical mean demand for the same zone, day of week, and 15-minute time slot.
    Fitted STRICTLY on the training set.
    """

    def __init__(self):
        self.lookup_table: Optional[pd.Series] = None
        self.zone_means: Optional[pd.Series] = None
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "HistoricalSeasonalBaseline":
        df = train_df.copy()
        if "time_slot_of_day" not in df.columns or "dayofweek" not in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            df["dayofweek"] = ts.dt.dayofweek
            df["time_slot_of_day"] = ts.dt.hour * 4 + ts.dt.minute // 15

        # Group by Zone x Day-of-Week x 15-min Slot
        self.lookup_table = (
            df.groupby(["PULocationID", "dayofweek", "time_slot_of_day"])["demand"]
            .mean()
        )
        self.zone_means = df.groupby("PULocationID")["demand"].mean()
        self.global_mean = float(df["demand"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.lookup_table is None:
            raise RuntimeError("Model must be fitted before predict.")

        df = df.copy()
        if "time_slot_of_day" not in df.columns or "dayofweek" not in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            df["dayofweek"] = ts.dt.dayofweek
            df["time_slot_of_day"] = ts.dt.hour * 4 + ts.dt.minute // 15

        # Map lookup
        keys = df.set_index(["PULocationID", "dayofweek", "time_slot_of_day"]).index
        preds = self.lookup_table.reindex(keys).to_numpy()

        # Fallback to zone mean or global mean if key not found
        if np.isnan(preds).any():
            nan_mask = np.isnan(preds)
            zone_fallback = df.loc[nan_mask, "PULocationID"].map(self.zone_means).to_numpy()
            preds[nan_mask] = np.where(np.isnan(zone_fallback), self.global_mean, zone_fallback)

        return np.maximum(0.0, preds)


class MovingAverageBaseline:
    """
    Baseline C: Moving average of previous k intervals (e.g., 4 intervals = 1 hour).
    Strictly uses information available before prediction time (shift(1).rolling(k).mean()).
    """

    def __init__(self, window: int = 4):
        self.window = window

    def fit(self, train_df: pd.DataFrame) -> "MovingAverageBaseline":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        col = f"rolling_mean_{self.window}"
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
        raise ValueError(
            f"DataFrame must contain '{col}' column. Ensure feature pipeline generated rolling window {self.window}."
        )
