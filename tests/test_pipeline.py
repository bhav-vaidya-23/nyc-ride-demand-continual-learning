"""
End-to-end pipeline validation and unit tests for demand_prediction.
"""
# Unit tests for demand_prediction

import numpy as np
import pandas as pd
from src.data_loader import inspect_raw_metadata
from src.features import (
    create_calendar_features,
    create_lag_and_rolling_features,
    get_feature_columns,
)
from src.baselines import (
    NaiveLastPeriodBaseline,
    HistoricalSeasonalBaseline,
    MovingAverageBaseline,
)
from src.metrics import calculate_metrics, build_comparison_table
from src.drift import calculate_psi, calculate_ks_test, detect_sustained_degradation
from src.continual import ModelRegistry


def test_metadata_inspection():
    meta = inspect_raw_metadata()
    assert meta["num_rows"] == 20405666
    assert "request_datetime" in meta["column_names"]
    assert "PULocationID" in meta["column_names"]


def test_features_and_baselines_synthetic():
    # Construct synthetic 15-min series for 2 zones over 100 intervals
    timestamps = pd.date_range("2025-01-01", periods=100, freq="15min")
    df_list = []
    for zone in [10, 20]:
        base = 20 if zone == 10 else 50
        noise = np.sin(np.linspace(0, 10, 100)) * 10
        demands = np.maximum(0, (base + noise).astype(int))
        df_list.append(pd.DataFrame({
            "timestamp": timestamps,
            "PULocationID": zone,
            "demand": demands,
        }))
    df = pd.concat(df_list).sort_values(["PULocationID", "timestamp"]).reset_index(drop=True)

    # Feature engineering test
    df_feat = create_calendar_features(df)
    assert "hour" in df_feat.columns
    assert "dayofweek" in df_feat.columns

    df_feat = create_lag_and_rolling_features(df_feat, lag_steps=[1, 2, 4], rolling_windows=[4])
    assert "lag_1" in df_feat.columns
    assert "rolling_mean_4" in df_feat.columns

    # Verify shift(1) guarantees no leakage: lag_1 must match previous demand
    for zone in [10, 20]:
        z_df = df_feat[df_feat["PULocationID"] == zone].reset_index(drop=True)
        assert np.isnan(z_df.loc[0, "lag_1"])
        assert z_df.loc[1, "lag_1"] == z_df.loc[0, "demand"]

    # Baseline tests
    train_slice = df_feat.iloc[:120]
    test_slice = df_feat.iloc[120:].dropna(subset=["lag_1", "rolling_mean_4"])

    naive = NaiveLastPeriodBaseline()
    naive_preds = naive.predict(test_slice)
    assert len(naive_preds) == len(test_slice)
    assert not np.isnan(naive_preds).any()

    seasonal = HistoricalSeasonalBaseline().fit(train_slice)
    seasonal_preds = seasonal.predict(test_slice)
    assert len(seasonal_preds) == len(test_slice)
    assert not np.isnan(seasonal_preds).any()

    ma = MovingAverageBaseline(window=4)
    ma_preds = ma.predict(test_slice)
    assert len(ma_preds) == len(test_slice)

    # Metrics test
    y_test = test_slice["demand"].to_numpy()
    res = build_comparison_table({
        "Naive (t-1)": naive_preds,
        "Seasonal": seasonal_preds,
        "Moving Average (1h)": ma_preds,
    }, y_test)
    assert "MAE" in res.columns
    assert len(res) == 3


def test_drift_and_degradation():
    ref = np.random.normal(50, 10, 1000)
    cur_stable = np.random.normal(50, 10, 1000)
    cur_drift = np.random.normal(70, 15, 1000)

    psi_stable = calculate_psi(ref, cur_stable)
    psi_drift = calculate_psi(ref, cur_drift)
    assert psi_stable < 0.15
    assert psi_drift > 0.25

    ks_res = calculate_ks_test(ref, cur_drift)
    assert ks_res["p_value"] < 0.01

    # Degradation detector test
    rolling_mae = pd.Series([10.0, 10.2, 10.1, 12.0, 12.5, 13.0, 12.8])
    has_deg, indices = detect_sustained_degradation(rolling_mae, baseline_mae=10.0, threshold_pct=0.15, consecutive_periods=3)
    assert has_deg is True
    assert len(indices) > 0


if __name__ == "__main__":
    print("Running unit tests...")
    test_metadata_inspection()
    print("Metadata test passed!")
    test_features_and_baselines_synthetic()
    print("Features and baselines test passed!")
    test_drift_and_degradation()
    print("Drift and degradation test passed!")
    print("All unit tests passed successfully!")
