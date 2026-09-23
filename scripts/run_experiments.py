"""
End-to-end execution of baseline evaluation, ML model training, stream simulation,
monitoring, and continual learning.
Saves model_v1, model_v2, model_registry.json, and stream_predictions.parquet.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from src.config import (
    MODELS_DIR,
    STREAM_PREDICTIONS_PARQUET,
    DEMAND_15MIN_PARQUET,
)
from src.data_loader import load_processed_demand, split_chronological
from src.features import build_feature_pipeline, get_feature_columns
from src.baselines import (
    NaiveLastPeriodBaseline,
    HistoricalSeasonalBaseline,
    MovingAverageBaseline,
)
from src.metrics import calculate_metrics, build_comparison_table, evaluate_subgroups
from src.drift import calculate_psi, calculate_ks_test, compute_rolling_metrics, detect_sustained_degradation
from src.continual import ModelRegistry, evaluate_champion_challenger


def run_all():
    print("=" * 80)
    print("STEP 3 & 4: LOADING 15-MINUTE DEMAND GRID & EVALUATING BASELINES")
    print("=" * 80)
    t0 = time.time()
    grid_df = load_processed_demand()
    print(f"Loaded {len(grid_df):,} grid records.")

    print("\nBuilding leak-free lag and rolling features...")
    feat_df = build_feature_pipeline(grid_df, drop_burn_in=True)
    feature_cols = get_feature_columns()
    print(f"Engineered {len(feature_cols)} features across {len(feat_df):,} records in {time.time() - t0:.2f}s.")

    # Chronological Split
    train_df, val_df, test_df = split_chronological(feat_df)
    print(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df):,} rows)")
    print(f"Val period:   {val_df['timestamp'].min()} to {val_df['timestamp'].max()} ({len(val_df):,} rows)")
    print(f"Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df):,} rows)")

    # Baseline predictions on Test set
    print("\nComputing Baseline forecasts on Test set...")
    naive_preds = NaiveLastPeriodBaseline().predict(test_df)
    seasonal_preds = HistoricalSeasonalBaseline().fit(train_df).predict(test_df)
    ma_preds = MovingAverageBaseline(window=4).predict(test_df)

    y_test = test_df["demand"].to_numpy()

    print("\n" + "=" * 80)
    print("STEP 5 & 6: TRAINING MACHINE LEARNING MODEL (LIGHTGBM) & FAIR BENCHMARKING")
    print("=" * 80)
    X_train, y_train = train_df[feature_cols], train_df["demand"].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["demand"].to_numpy()
    X_test = test_df[feature_cols]

    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.08,
        "num_leaves": 63,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    t_train = time.time()
    ml_model = lgb.LGBMRegressor(**lgb_params)
    ml_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    print(f"ML Model trained in {time.time() - t_train:.2f}s.")

    val_preds = ml_model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_preds)
    print(f"Validation Performance: MAE={val_metrics['MAE']}, RMSE={val_metrics['RMSE']}")

    ml_test_preds = np.maximum(0.0, ml_model.predict(X_test))

    # DECISIVE BENCHMARK COMPARISON TABLE
    models_dict = {
        "Baseline A: Naive (t-1)": naive_preds,
        "Baseline B: Historical Seasonal": seasonal_preds,
        "Baseline C: Moving Average (1h)": ma_preds,
        "Machine Learning (LightGBM)": ml_test_preds,
    }
    comp_df = build_comparison_table(models_dict, y_test, baseline_key="Baseline A: Naive (t-1)")
    print("\n" + "*" * 80)
    print("                     DECISIVE BENCHMARK COMPARISON TABLE                      ")
    print("*" * 80)
    print(comp_df.to_string(index=False))

    # Subgroups
    top_zones = train_df.groupby("PULocationID")["demand"].sum().sort_values(ascending=False).head(52).index.tolist()
    subgroup_eval = evaluate_subgroups(test_df, ml_test_preds, high_demand_zones=top_zones)
    print("\nML Performance Across Subgroups:")
    print(pd.DataFrame(subgroup_eval).T)

    # Save model_v1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_v1_path = MODELS_DIR / "model_v1.joblib"
    joblib.dump(ml_model, model_v1_path)

    registry = ModelRegistry()
    registry.register_model(
        version="model_v1",
        model_type="LightGBM Regressor",
        training_period=(str(train_df["timestamp"].min()), str(train_df["timestamp"].max())),
        features=feature_cols,
        val_metrics=val_metrics,
        status="champion",
        reason="Initial champion trained on Jan 8-20, validated on Jan 21-25",
        artifact_path=str(model_v1_path),
    )
    print(f"\nSaved Champion to {model_v1_path} and registered in model_registry.json.")

    print("\n" + "=" * 80)
    print("STEP 8 & 9: REAL-TIME STREAM SIMULATION & PREDICTION STORAGE")
    print("=" * 80)
    print("Replaying historical test stream (Jan 26-31)...")
    unique_ts = sorted(test_df["timestamp"].unique())
    records = []
    t_sim = time.time()

    for ts in unique_ts:
        interval_data = test_df[test_df["timestamp"] == ts]
        preds = np.maximum(0.0, ml_model.predict(interval_data[feature_cols]))
        actuals = interval_data["demand"].to_numpy()
        zones = interval_data["PULocationID"].to_numpy()
        errors = preds - actuals
        abs_errors = np.abs(errors)

        for z, p, a, e, ae in zip(zones, preds, actuals, errors, abs_errors):
            records.append({
                "prediction_timestamp": ts,
                "zone": int(z),
                "prediction_horizon": "15min",
                "predicted_demand": round(float(p), 2),
                "actual_demand": int(a),
                "error": round(float(e), 2),
                "absolute_error": round(float(ae), 2),
                "model_version": "model_v1",
            })

    pred_df = pd.DataFrame(records)
    pred_df.to_parquet(STREAM_PREDICTIONS_PARQUET, index=False, engine="pyarrow", compression="snappy")
    print(f"Logged {len(pred_df):,} streaming predictions in {time.time() - t_sim:.2f}s.")
    print(f"Saved to {STREAM_PREDICTIONS_PARQUET}.")

    print("\n" + "=" * 80)
    print("STEP 10 & 11: PERFORMANCE MONITORING & DRIFT DETECTION")
    print("=" * 80)
    rolling_summary = compute_rolling_metrics(pred_df, window_size=96)
    print("Rolling 24-hour summary sample:")
    print(rolling_summary.tail(5)[["prediction_timestamp", "rolling_mae", "rolling_rmse", "rolling_bias"]])

    # Drift checks
    print("\nComputing Feature Data Drift (PSI & KS-test):")
    for feat in ["lag_1", "lag_96", "rolling_mean_4"]:
        psi_val = calculate_psi(train_df[feat].to_numpy(), test_df[feat].to_numpy())
        ks_res = calculate_ks_test(train_df[feat].to_numpy(), test_df[feat].to_numpy())
        print(f"  - {feat:<16}: PSI = {psi_val:.4f}, KS Stat = {ks_res['ks_statistic']:.4f} (p={ks_res['p_value']:.4e})")

    baseline_mae = val_metrics["MAE"]
    has_deg, deg_idx = detect_sustained_degradation(rolling_summary["rolling_mae"], baseline_mae, threshold_pct=0.15)
    print(f"\nSustained degradation check: {has_deg}")

    print("\n" + "=" * 80)
    print("STEP 12: CONTINUAL LEARNING / CHAMPION-CHALLENGER UPDATE WORKFLOW")
    print("=" * 80)
    print("Training candidate model_v2 with expanded sliding window (Jan 8-25)...")
    challenger_mask = (feat_df["timestamp"] >= "2025-01-08") & (feat_df["timestamp"] <= "2025-01-25 23:59:59")
    chall_train_df = feat_df[challenger_mask]

    holdout_mask = (feat_df["timestamp"] >= "2025-01-26") & (feat_df["timestamp"] <= "2025-01-28 23:59:59")
    holdout_df = feat_df[holdout_mask]

    challenger_model = lgb.LGBMRegressor(**lgb_params)
    challenger_model.fit(chall_train_df[feature_cols], chall_train_df["demand"].to_numpy())

    promoted, gate_res = evaluate_champion_challenger(
        champion_model=ml_model,
        challenger_model=challenger_model,
        holdout_df=holdout_df,
        feature_cols=feature_cols,
        high_demand_zones=top_zones,
        min_improvement=0.03,
    )

    print("\nChampion vs Challenger Evaluation Gate:")
    for k, v in gate_res.items():
        print(f"  {k:<28}: {v}")

    model_v2_path = MODELS_DIR / "model_v2.joblib"
    joblib.dump(challenger_model, model_v2_path)

    if promoted:
        print("\n--> PROMOTION SUCCESS: Promoting model_v2 to Champion!")
        registry.register_model(
            version="model_v2",
            model_type="LightGBM Regressor (Retrained)",
            training_period=("2025-01-08", "2025-01-25"),
            features=feature_cols,
            val_metrics={"MAE": gate_res["challenger_mae"]},
            status="champion",
            reason=f"Promoted over model_v1: +{gate_res['pct_overall_improvement']}% overall MAE improvement",
            artifact_path=str(model_v2_path),
        )
        registry.promote_to_champion("model_v2", reason="Beats model_v1 on holdout validation gate")
    else:
        print("\n--> PROMOTION REJECTED: Retaining model_v1 as Champion.")
        registry.register_model(
            version="model_v2",
            model_type="LightGBM Regressor (Retrained)",
            training_period=("2025-01-08", "2025-01-25"),
            features=feature_cols,
            val_metrics={"MAE": gate_res["challenger_mae"]},
            status="rejected",
            reason="Did not achieve required improvement margin over champion",
            artifact_path=str(model_v2_path),
        )

    print("\nUpdated Model Registry Lineage:")
    print(registry.get_lineage_table().to_string(index=False))
    print("\nAll experiment steps completed successfully in", round(time.time() - t0, 2), "s.")


if __name__ == "__main__":
    run_all()
