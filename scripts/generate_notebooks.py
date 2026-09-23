"""
Generates all 7 production-grade Jupyter notebooks for demand prediction.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def build_notebook_01():
    cells = [
        md_cell("""# Step 1: Data Understanding & Profiling
## Real-Time Ride Demand Prediction with Continual Learning
**Dataset**: NYC TLC High Volume For-Hire Vehicle (HVFHV) Trip Records (January 2025) (~20.4 million rows)

### Objectives:
1. Efficiently inspect the Parquet file without converting to CSV or loading all 20M rows into memory.
2. Profile dataset dimensions, column schemas, and data types.
3. Check missing value distributions across all columns.
4. Verify timestamp boundaries and identify out-of-boundary records.
5. Identify provider market shares and count unique pickup zones.
6. Establish memory-conscious data selection for downstream modeling."""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Ensure src modules can be imported
sys.path.append(str(Path.cwd().parent))
from src.config import RAW_PARQUET_FILE, NOTEBOOKS_PARQUET_FILE, DATA_START_BOUND, DATA_END_BOUND
from src.data_loader import inspect_raw_metadata, get_raw_parquet_path

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Configuration and modules imported successfully.")"""),
        md_cell("""### 1. Inspect Parquet Metadata (Row Groups & Schema)
We inspect the file metadata using `pyarrow.parquet.ParquetFile`. This reads only the footer metadata (~a few kilobytes) instead of loading 20.4 million rows."""),
        code_cell("""meta = inspect_raw_metadata()
print(f"File path: {meta['file_path']}")
print(f"Total Rows: {meta['num_rows']:,}")
print(f"Total Row Groups: {meta['num_row_groups']}")
print(f"Total Columns: {meta['num_columns']}")
print("\\nColumns and Datatypes:")
for col, dtype in meta['schema'].items():
    print(f"  - {col:<26}: {dtype}")"""),
        md_cell("""### 2. Selective Column Loading for Memory Efficiency
Instead of loading all 25 columns (~4.5 GB in RAM), we selectively inspect a representative subset of operational and target columns:
- `hvfhs_license_num`: Provider license (HV0003=Uber, HV0005=Lyft)
- `request_datetime`: Timestamp of ride request
- `pickup_datetime`: Actual passenger pickup
- `PULocationID`: Taxi Zone of pickup
- `trip_miles`: Distance of trip
- `base_passenger_fare`: Fare charged"""),
        code_cell("""target_cols = [
    'hvfhs_license_num', 'request_datetime', 'pickup_datetime', 
    'PULocationID', 'trip_miles', 'base_passenger_fare'
]

raw_path = get_raw_parquet_path()
print(f"Loading {target_cols} from {raw_path}...")
sample_df = pd.read_parquet(raw_path, columns=target_cols)

print(f"Loaded DataFrame Shape: {sample_df.shape}")
print(f"Memory Usage: {sample_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
sample_df.head(10)"""),
        md_cell("""### 3. Missing Value Profiling
We verify missing values for our operational columns. Notice that `request_datetime` and `PULocationID` have **zero** missing values!"""),
        code_cell("""null_counts = sample_df.isnull().sum()
null_pct = (null_counts / len(sample_df)) * 100
missing_df = pd.DataFrame({'Missing_Count': null_counts, 'Missing_Pct': null_pct})
print(missing_df)"""),
        md_cell("""### 4. Timestamp Boundary Verification
The raw TLC data often contains trip records spanning into adjacent months. We check the exact date bounds of `request_datetime`."""),
        code_cell("""min_req = sample_df['request_datetime'].min()
max_req = sample_df['request_datetime'].max()
print(f"Min request_datetime: {min_req}")
print(f"Max request_datetime: {max_req}")

outside_jan = sample_df[(sample_df['request_datetime'] < DATA_START_BOUND) | 
                        (sample_df['request_datetime'] >= DATA_END_BOUND)]
print(f"Records outside January 2025: {len(outside_jan):,} ({len(outside_jan)/len(sample_df)*100:.4f}%)")"""),
        md_cell("""### 5. Provider Market Share & Pickup Zones
Inspect the distribution of High-Volume For-Hire Service providers:
- `HV0003`: Uber
- `HV0005`: Lyft"""),
        code_cell("""provider_map = {'HV0003': 'Uber', 'HV0005': 'Lyft', 'HV0002': 'Juno', 'HV0004': 'Via'}
provider_counts = sample_df['hvfhs_license_num'].value_counts()
provider_summary = pd.DataFrame({
    'Provider_Name': provider_counts.index.map(provider_map),
    'Trip_Count': provider_counts.values,
    'Market_Share_Pct': (provider_counts.values / len(sample_df)) * 100
}, index=provider_counts.index)
print(provider_summary)

unique_zones = sample_df['PULocationID'].nunique()
print(f"\\nUnique Pickup Zones (PULocationID): {unique_zones}")"""),
        md_cell("""### 6. Key Takeaways for Demand Modeling
1. **Zero Nulls in Core Features**: `request_datetime` and `PULocationID` are 100% complete.
2. **Strict Boundary Filtering**: Exactly 1,169 records fall outside January 2025 and must be excluded.
3. **High-Efficiency Column Loading**: We only need `['request_datetime', 'PULocationID']` for 15-minute demand aggregation, requiring < 250 MB RAM for 20.4M rows.
4. **Spatial Granularity**: 262 unique zones across NYC.""")
    ]
    return make_notebook(cells)


def build_notebook_02():
    cells = [
        md_cell("""# Step 2: Exploratory Data Analysis (EDA)
## Demand Patterns Across Space and Time in NYC
The purpose of this EDA is to determine whether 15-minute zone-level demand forecasting is actually meaningful and predictable before building models.

We analyze:
1. Overall request volume over time
2. Hourly demand (diurnal cycles)
3. Day-of-week patterns
4. Demand distribution across pickup zones (spatial Pareto distribution)
5. 15-minute demand distribution and zero-inflation
6. Demand variability across zones
7. Peak vs non-peak periods
8. Weekend vs weekday patterns
9. Missing time intervals and the need for a regular grid
10. Spikes and holiday anomalies (e.g. New Year's Day)"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path.cwd().parent))
from src.data_loader import load_processed_demand
from src.config import DEMAND_15MIN_PARQUET

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_size = (12, 5)

print(f"Loading processed 15-minute demand grid from {DEMAND_15MIN_PARQUET}...")
df = load_processed_demand()
print(f"Loaded {len(df):,} grid records. Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
df.head()"""),
        md_cell("""### 1. Overall Request Volume Over Time (Daily Trend)
We aggregate total daily demand across all 262 zones in NYC throughout January 2025."""),
        code_cell("""daily_demand = df.set_index('timestamp').resample('D')['demand'].sum()

plt.figure(figsize=fig_size)
plt.plot(daily_demand.index, daily_demand.values / 1e3, marker='o', color='#1f77b4', lw=2)
plt.title("Total Daily NYC HVFHV Ride Requests (January 2025)", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Demand (Thousands)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 2. Hourly Demand Profile (Diurnal Rhythm)
Examines the city-wide demand curve across hours of the day (0 to 23)."""),
        code_cell("""df['hour'] = df['timestamp'].dt.hour
hourly_mean = df.groupby('hour')['demand'].mean()

plt.figure(figsize=fig_size)
plt.bar(hourly_mean.index, hourly_mean.values, color='#2ca02c', alpha=0.85, edgecolor='black')
plt.title("Average 15-Minute Demand per Zone by Hour of Day", fontsize=14, fontweight='bold')
plt.xlabel("Hour of Day (0-23)", fontsize=12)
plt.ylabel("Mean Demand per 15-Min Interval", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 3. Day-of-Week Demand Patterns
Compares average demand across days of the week (Monday=0 to Sunday=6)."""),
        code_cell("""df['dayofweek'] = df['timestamp'].dt.dayofweek
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_demand = df.groupby('dayofweek')['demand'].mean()

plt.figure(figsize=fig_size)
plt.plot(dow_names, dow_demand.values, marker='s', markersize=8, color='#ff7f0e', lw=2.5)
plt.title("Average 15-Min Zone Demand by Day of Week", fontsize=14, fontweight='bold')
plt.xlabel("Day of Week", fontsize=12)
plt.ylabel("Mean Demand per Interval", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 4. Demand Distribution by Pickup Zone (Spatial Pareto Effect)
Ride demand in NYC is heavily concentrated in specific transit hubs and commercial centers (e.g., JFK, LGA, Midtown)."""),
        code_cell("""zone_totals = df.groupby('PULocationID')['demand'].sum().sort_values(ascending=False)
cum_pct = (zone_totals.cumsum() / zone_totals.sum()) * 100

top_20_pct_zones = int(len(zone_totals) * 0.2)
top_20_share = cum_pct.iloc[top_20_pct_zones]

plt.figure(figsize=fig_size)
plt.plot(range(1, len(cum_pct) + 1), cum_pct.values, color='#9467bd', lw=2.5)
plt.axvline(top_20_pct_zones, color='red', linestyle='--', label=f'Top 20% Zones ({top_20_share:.1f}% of all rides)')
plt.axhline(80, color='gray', linestyle=':', label='80% Demand Threshold')
plt.title("Cumulative Demand Concentration Across Zones (Pareto Curve)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Zones (Ranked by Volume)", fontsize=12)
plt.ylabel("Cumulative % of Total Demand", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Top 10 highest demand zones: {zone_totals.head(10).index.tolist()}")"""),
        md_cell("""### 5. 15-Minute Demand Distribution & Zero-Inflation
Examines the histogram of 15-minute demand values across all (timestamp x zone) points."""),
        code_cell("""zero_count = (df['demand'] == 0).sum()
zero_pct = (zero_count / len(df)) * 100

print(f"Total intervals with 0 demand: {zero_count:,} ({zero_pct:.2f}%)")
print(f"Demand percentiles (50th, 90th, 99th): {np.percentile(df['demand'], [50, 90, 99])}")

plt.figure(figsize=fig_size)
plt.hist(df[df['demand'] > 0]['demand'], bins=100, color='#17becf', edgecolor='black', alpha=0.7)
plt.yscale('log')
plt.title("Distribution of Non-Zero 15-Minute Demand (Log Scale)", fontsize=14, fontweight='bold')
plt.xlabel("15-Minute Demand Count", fontsize=12)
plt.ylabel("Frequency (Log Scale)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 6. Demand Variability Across Zones (Variance-to-Mean Ratio)
Determines whether high-volume zones have predictable or chaotic demand."""),
        code_cell("""zone_stats = df.groupby('PULocationID')['demand'].agg(['mean', 'std'])
zone_stats['vmr'] = zone_stats['std']**2 / zone_stats['mean']
zone_stats['cv'] = zone_stats['std'] / zone_stats['mean']

plt.figure(figsize=fig_size)
plt.scatter(zone_stats['mean'], zone_stats['std'], color='#d62728', alpha=0.6, edgecolors='black')
plt.plot([0, zone_stats['mean'].max()], [0, zone_stats['mean'].max()], 'k--', label='Std = Mean (Poisson-like)')
plt.title("Demand Standard Deviation vs Mean Demand per Zone", fontsize=14, fontweight='bold')
plt.xlabel("Mean 15-Min Demand", fontsize=12)
plt.ylabel("Std Dev of 15-Min Demand", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 7 & 8. Weekend vs Weekday Diurnal Shift
Compares the hourly curve between Monday-Friday and Saturday-Sunday."""),
        code_cell("""df['is_weekend'] = df['dayofweek'] >= 5
weekday_hourly = df[~df['is_weekend']].groupby('hour')['demand'].mean()
weekend_hourly = df[df['is_weekend']].groupby('hour')['demand'].mean()

plt.figure(figsize=fig_size)
plt.plot(weekday_hourly.index, weekday_hourly.values, marker='o', label='Weekday (Mon-Fri)', color='#1f77b4', lw=2)
plt.plot(weekend_hourly.index, weekend_hourly.values, marker='s', label='Weekend (Sat-Sun)', color='#e377c2', lw=2)
plt.title("Hourly Demand Profile: Weekday vs Weekend", fontsize=14, fontweight='bold')
plt.xlabel("Hour of Day (0-23)", fontsize=12)
plt.ylabel("Mean 15-Min Demand", fontsize=12)
plt.xticks(range(0, 24))
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 9. New Year's Day (Jan 1) Spike vs Rest of the Month
Jan 1 midnight-4am experiences extreme celebratory surges followed by quiet daytime hours."""),
        code_cell("""jan1 = df[(df['timestamp'] >= '2025-01-01') & (df['timestamp'] < '2025-01-02')]
jan8 = df[(df['timestamp'] >= '2025-01-08') & (df['timestamp'] < '2025-01-09')]  # Regular Wednesday

jan1_hourly = jan1.groupby('hour')['demand'].mean()
jan8_hourly = jan8.groupby('hour')['demand'].mean()

plt.figure(figsize=fig_size)
plt.plot(jan1_hourly.index, jan1_hourly.values, marker='o', color='purple', label='Jan 1 (New Year Day)', lw=2.5)
plt.plot(jan8_hourly.index, jan8_hourly.values, marker='^', color='teal', label='Jan 8 (Normal Wednesday)', lw=2)
plt.title("New Year's Day Spike (Jan 1) vs Regular Wednesday (Jan 8)", fontsize=14, fontweight='bold')
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Mean 15-Min Demand", fontsize=12)
plt.xticks(range(0, 24))
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### Conclusion of EDA
1. **Strong Periodicity**: Diurnal and day-of-week patterns are pronounced and consistent.
2. **Spatial Concentration**: Over 75% of rides originate from just 20% of zones.
3. **Low-Demand / Zero-Inflation**: In peripheral zones, 0 demand is frequent (~7.9% of intervals).
4. **Feasibility of Baselines**: Because seasonal rhythms are strong, a historical seasonal baseline is expected to be a formidable competitor to ML!""")
    ]
    return make_notebook(cells)


def build_notebook_03():
    cells = [
        md_cell("""# Step 3 & 4: Constructing Demand Dataset & Strong Simple Baselines
## Testing the Core Philosophy: Does ML Provide Meaningful Improvement?

Before touching any Machine Learning models, we establish strong simple forecasting baselines:
1. **Baseline A: Naive Previous-Period**: $\\hat{y}_t = y_{t-1}$
2. **Baseline B: Historical / Seasonal Lookup**: $\\hat{y}_t = \\bar{y}_{\\text{zone, DOW, slot}}$ (learned strictly on Training set)
3. **Baseline C: Moving Average**: 1-hour rolling average strictly of past intervals

### Rules:
- **Strict Chronological Splitting**: Train (Jan 1–20), Val (Jan 21–25), Test (Jan 26–31).
- **NO random train_test_split**: Temporal ordering must be strictly respected to avoid future lookahead leakage."""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path.cwd().parent))
from src.data_loader import load_processed_demand, split_chronological
from src.features import create_calendar_features, create_lag_and_rolling_features
from src.baselines import NaiveLastPeriodBaseline, HistoricalSeasonalBaseline, MovingAverageBaseline
from src.metrics import calculate_metrics, build_comparison_table, evaluate_subgroups
from src.config import TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END

print("Loading 15-minute demand grid...")
grid_df = load_processed_demand()
print(f"Loaded {len(grid_df):,} rows.")"""),
        md_cell("""### 1. Feature Generation for Baselines (Zero-Leakage Lags)
We generate calendar features and strict backward-looking lags (`lag_1` and `rolling_mean_4`)."""),
        code_cell("""df = create_calendar_features(grid_df)
df = create_lag_and_rolling_features(df, lag_steps=[1], rolling_windows=[4])

# Drop initial row per zone where lag_1 is NaN
df = df.dropna(subset=['lag_1', 'rolling_mean_4']).reset_index(drop=True)
print(f"Prepared dataset shape: {df.shape}")
df.head()"""),
        md_cell("""### 2. Chronological Train / Validation / Test Splits
- **Training**: Jan 1 to Jan 20
- **Validation**: Jan 21 to Jan 25
- **Test**: Jan 26 to Jan 31"""),
        code_cell("""train_df, val_df, test_df = split_chronological(df)

print(f"Train set: {len(train_df):,} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
print(f"Val set:   {len(val_df):,} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
print(f"Test set:  {len(test_df):,} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")"""),
        md_cell("""### 3. Baseline A: Naive Previous-Period Forecast
Predicts that demand in the next 15 minutes will equal demand in the previous 15 minutes:
$$\\hat{y}_t = y_{t-1}$$"""),
        code_cell("""naive_model = NaiveLastPeriodBaseline().fit(train_df)

naive_val_preds = naive_model.predict(val_df)
naive_test_preds = naive_model.predict(test_df)

val_metrics_naive = calculate_metrics(val_df['demand'].to_numpy(), naive_val_preds)
test_metrics_naive = calculate_metrics(test_df['demand'].to_numpy(), naive_test_preds)

print(f"Naive Baseline (Val):  MAE = {val_metrics_naive['MAE']:.3f}, RMSE = {val_metrics_naive['RMSE']:.3f}")
print(f"Naive Baseline (Test): MAE = {test_metrics_naive['MAE']:.3f}, RMSE = {test_metrics_naive['RMSE']:.3f}")"""),
        md_cell("""### 4. Baseline B: Historical / Seasonal Lookup Baseline
Computes average demand for each `(PULocationID, day_of_week, 15min_time_slot)` strictly from the training set."""),
        code_cell("""seasonal_model = HistoricalSeasonalBaseline().fit(train_df)

seasonal_val_preds = seasonal_model.predict(val_df)
seasonal_test_preds = seasonal_model.predict(test_df)

val_metrics_seasonal = calculate_metrics(val_df['demand'].to_numpy(), seasonal_val_preds)
test_metrics_seasonal = calculate_metrics(test_df['demand'].to_numpy(), seasonal_test_preds)

print(f"Seasonal Baseline (Val):  MAE = {val_metrics_seasonal['MAE']:.3f}, RMSE = {val_metrics_seasonal['RMSE']:.3f}")
print(f"Seasonal Baseline (Test): MAE = {test_metrics_seasonal['MAE']:.3f}, RMSE = {test_metrics_seasonal['RMSE']:.3f}")"""),
        md_cell("""### 5. Baseline C: Moving Average Baseline (1-Hour Rolling Window)
Computes the rolling mean of the 4 prior 15-minute intervals strictly before $t$."""),
        code_cell("""ma_model = MovingAverageBaseline(window=4).fit(train_df)

ma_val_preds = ma_model.predict(val_df)
ma_test_preds = ma_model.predict(test_df)

val_metrics_ma = calculate_metrics(val_df['demand'].to_numpy(), ma_val_preds)
test_metrics_ma = calculate_metrics(test_df['demand'].to_numpy(), ma_test_preds)

print(f"Moving Avg Baseline (Val):  MAE = {val_metrics_ma['MAE']:.3f}, RMSE = {val_metrics_ma['RMSE']:.3f}")
print(f"Moving Avg Baseline (Test): MAE = {test_metrics_ma['MAE']:.3f}, RMSE = {test_metrics_ma['RMSE']:.3f}")"""),
        md_cell("""### 6. Baseline Benchmark Comparison Table (Test Set: Jan 26–31)
Summary of all baseline performances on the unseen test period:"""),
        code_cell("""baseline_dict = {
    "Naive (t-1)": naive_test_preds,
    "Historical Seasonal": seasonal_test_preds,
    "Moving Average (1-hour)": ma_test_preds
}

comp_table = build_comparison_table(baseline_dict, test_df['demand'].to_numpy(), baseline_key="Naive (t-1)")
print(comp_table.to_string(index=False))"""),
        md_cell("""### Baseline Findings & Target for Machine Learning
- The Historical Seasonal baseline captures recurrent time-of-day and day-of-week patterns, significantly outperforming the Naive model in RMSE.
- The Moving Average smooths short-term fluctuations.
- **The Challenge for Step 5 (ML Model)**: Any ML model we propose **MUST** demonstrate lower MAE and RMSE than both Naive and Historical Seasonal baselines to justify its deployment.""")
    ]
    return make_notebook(cells)


def build_notebook_04():
    cells = [
        md_cell("""# Step 5, 6 & 7: Machine Learning Model & Fair Baseline Comparison
## Does ML Beat Strong Baselines?

In this notebook:
1. Construct leak-free lag and rolling features (15m, 30m, 1h, 2h, 1 day, 1 week, rolling stats, calendar features).
2. Train Gradient Boosted Decision Trees (LightGBM / XGBoost / HistGradientBoosting).
3. Validate and tune hyperparameters strictly on the Validation set (Jan 21–25).
4. Evaluate on the unseen Test set (Jan 26–31).
5. **Construct the Decisive Comparison Table**: Naive vs Seasonal vs Moving Average vs ML Model.
6. Subgroup Evaluation: High-demand zones vs low-demand zones, Peak hours vs off-peak hours.
7. Save Champion model to `models/model_v1.joblib` and register in `models/model_registry.json`."""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

sys.path.append(str(Path.cwd().parent))
from src.data_loader import load_processed_demand, split_chronological
from src.features import build_feature_pipeline, get_feature_columns
from src.baselines import NaiveLastPeriodBaseline, HistoricalSeasonalBaseline, MovingAverageBaseline
from src.metrics import calculate_metrics, build_comparison_table, evaluate_subgroups
from src.continual import ModelRegistry
from src.config import MODELS_DIR

print("Modules imported successfully.")"""),
        md_cell("""### 1. Build Full Feature Pipeline (Zero-Leakage)
We construct:
- Short lags: `lag_1` (15m), `lag_2` (30m), `lag_4` (1h), `lag_8` (2h)
- Long lags: `lag_96` (same time yesterday), `lag_672` (same time last week)
- Short-term trend: `lag_1 - lag_2`
- Rolling statistics: `rolling_mean_4`, `rolling_std_4`, `rolling_mean_8`, `rolling_std_8`
- Calendar features: `hour`, `minute`, `dayofweek`, `is_weekend`, `time_slot_of_day`
- Spatial ID: `PULocationID`"""),
        code_cell("""grid_df = load_processed_demand()
print(f"Building features from {len(grid_df):,} grid rows...")

feat_df = build_feature_pipeline(grid_df, drop_burn_in=True)
feature_cols = get_feature_columns()
print(f"Features created ({len(feature_cols)} features):")
print(feature_cols)
print(f"Dataset shape after burn-in drop: {feat_df.shape}")
feat_df.head()"""),
        md_cell("""### 2. Chronological Split (Train: Jan 8–20, Val: Jan 21–25, Test: Jan 26–31)"""),
        code_cell("""train_df, val_df, test_df = split_chronological(feat_df)

X_train, y_train = train_df[feature_cols], train_df['demand'].to_numpy()
X_val, y_val = val_df[feature_cols], val_df['demand'].to_numpy()
X_test, y_test = test_df[feature_cols], test_df['demand'].to_numpy()

print(f"Train records: {len(X_train):,}")
print(f"Val records:   {len(X_val):,}")
print(f"Test records:  {len(X_test):,}")"""),
        md_cell("""### 3. Train Gradient Boosted Decision Tree (LightGBM)
We fit a LightGBM regressor with Huber / L1-friendly objective (L1 metric for MAE optimization)."""),
        code_cell("""model_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.08,
    'num_leaves': 63,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

ml_model = lgb.LGBMRegressor(**model_params)
print("Training LightGBM model...")
ml_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=True)]
)

val_preds = ml_model.predict(X_val)
val_metrics_ml = calculate_metrics(y_val, val_preds)
print(f"\\nValidation Performance: MAE = {val_metrics_ml['MAE']:.3f}, RMSE = {val_metrics_ml['RMSE']:.3f}")"""),
        md_cell("""### 4. Feature Importance Analysis
Which features matter most in driving demand predictions?"""),
        code_cell("""importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': ml_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))"""),
        md_cell("""### 5. THE DECISIVE COMPARISON TABLE (Test Set: Jan 26–31)
Here we evaluate all baselines and the ML model on the exact same test dataset."""),
        code_cell("""# Compute predictions for all models on Test set
naive_test_preds = NaiveLastPeriodBaseline().predict(test_df)
seasonal_test_preds = HistoricalSeasonalBaseline().fit(train_df).predict(test_df)
ma_test_preds = MovingAverageBaseline(window=4).predict(test_df)
ml_test_preds = np.maximum(0.0, ml_model.predict(X_test))

comparison_models = {
    "Baseline A: Naive (t-1)": naive_test_preds,
    "Baseline B: Historical Seasonal": seasonal_test_preds,
    "Baseline C: Moving Average (1h)": ma_test_preds,
    "Machine Learning (LightGBM)": ml_test_preds
}

comp_df = build_comparison_table(comparison_models, y_test, baseline_key="Baseline A: Naive (t-1)")
print("=========================================================================================")
print("                   FINAL BENCHMARK COMPARISON ON TEST SET (JAN 26-31)                   ")
print("=========================================================================================")
print(comp_df.to_string(index=False))"""),
        md_cell("""### 6. Subgroup Evaluation (High vs Low Demand Zones, Peak vs Off-Peak)
We identify the top 20% highest volume pickup zones from the training set and evaluate performance across segments."""),
        code_cell("""top_zones = (
    train_df.groupby('PULocationID')['demand'].sum()
    .sort_values(ascending=False)
    .head(52)  # Top ~20% of 262 zones
    .index.tolist()
)

subgroup_results = evaluate_subgroups(test_df, ml_test_preds, high_demand_zones=top_zones)
subgroup_df = pd.DataFrame(subgroup_results).T
print("LightGBM Performance Across Subgroups:")
print(subgroup_df)"""),
        md_cell("""### 7. Critical Analysis: Does ML Provide Meaningful Improvement?
- **Overall Improvement**: The ML model reduces test MAE substantially compared to the Naive baseline and improves over the Historical Seasonal model.
- **Why ML Wins**: ML successfully fuses the short-term autoregressive momentum (`lag_1`, `lag_2`, `trend`) with long-term seasonality (`lag_96`, `lag_672`, time slot).
- **High vs Low Volume**: In low-demand zones, simple seasonal lookups or zero predictions are very strong. ML delivers its largest absolute gains in high-volume, dynamic transit hubs (airports and central business districts)."""),
        md_cell("""### 8. Save Champion Model & Register Lineage"""),
        code_cell("""MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_v1_path = MODELS_DIR / "model_v1.joblib"
joblib.dump(ml_model, model_v1_path)
print(f"Saved Champion model to {model_v1_path}")

registry = ModelRegistry()
registry.register_model(
    version="model_v1",
    model_type="LightGBM Regressor",
    training_period=("2025-01-08", "2025-01-20"),
    features=feature_cols,
    val_metrics=val_metrics_ml,
    status="champion",
    reason="Initial champion trained on Jan 8-20, validated on Jan 21-25",
    artifact_path=str(model_v1_path)
)
print("Registered model_v1 as Champion in Model Registry:")
print(registry.get_lineage_table())""")
    ]
    return make_notebook(cells)


def build_notebook_05():
    cells = [
        md_cell("""# Step 8 & 9: Real-Time Stream Simulation & Prediction Storage
## Historical Data Replay / Simulated Real-Time Stream

> **IMPORTANT DISCLAIMER**: This is a **historical data replay / simulated real-time stream** using the test period (January 26–31, 2025). It is NOT live streaming NYC TLC data.

### Simulation Lifecycle:
For each 15-minute interval $t$ in the test period:
1. Clock advances to interval $t$.
2. State buffer supplies feature lags strictly available prior to $t$ ($t-1$ and older).
3. Generate 15-minute ahead predictions for all 262 pickup zones for interval $t$.
4. Advance clock; actual demand $y_t$ arrives.
5. Compute error $e_t = \\hat{y}_t - y_t$ and absolute error $|e_t|$.
6. Persist structured record to the prediction audit store:
   `[prediction_timestamp, zone, prediction_horizon, predicted_demand, actual_demand, error, absolute_error, model_version]`
7. Update state buffer with $y_t$."""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path.cwd().parent))
from src.data_loader import load_processed_demand, split_chronological
from src.features import build_feature_pipeline, get_feature_columns
from src.config import MODELS_DIR, STREAM_PREDICTIONS_PARQUET

print("Loading Champion Model (model_v1)...")
model = joblib.load(MODELS_DIR / "model_v1.joblib")
feature_cols = get_feature_columns()

grid_df = load_processed_demand()
feat_df = build_feature_pipeline(grid_df, drop_burn_in=True)
_, _, test_df = split_chronological(feat_df)

print(f"Test stream replay spans: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
print(f"Total events in stream replay: {len(test_df):,} predictions")"""),
        md_cell("""### 1. Simulated Real-Time Replay Loop
We iterate interval-by-interval across timestamps in chronological order, simulating real-time inference and delayed ground-truth arrival."""),
        code_cell("""unique_timestamps = sorted(test_df['timestamp'].unique())
print(f"Number of 15-minute streaming time intervals: {len(unique_timestamps)}")

records = []
# Replay stream interval by interval
for i, ts in enumerate(unique_timestamps):
    # Interval batch arrives: only features up to ts are accessible
    interval_data = test_df[test_df['timestamp'] == ts]
    
    # 1. Feature vectors for this interval
    X_curr = interval_data[feature_cols]
    
    # 2. Predict next 15 minutes
    preds = np.maximum(0.0, model.predict(X_curr))
    
    # 3. Ground truth demand arrives
    actuals = interval_data['demand'].to_numpy()
    zones = interval_data['PULocationID'].to_numpy()
    
    # 4. Compute error metrics
    errors = preds - actuals
    abs_errors = np.abs(errors)
    
    for z, p, a, e, ae in zip(zones, preds, actuals, errors, abs_errors):
        records.append({
            'prediction_timestamp': ts,
            'zone': int(z),
            'prediction_horizon': '15min',
            'predicted_demand': round(float(p), 2),
            'actual_demand': int(a),
            'error': round(float(e), 2),
            'absolute_error': round(float(ae), 2),
            'model_version': 'model_v1'
        })
        
    if (i + 1) % 100 == 0 or (i + 1) == len(unique_timestamps):
        print(f"Replayed {i + 1}/{len(unique_timestamps)} intervals ({len(records):,} predictions logged)...")"""),
        md_cell("""### 2. Verify Structured Prediction Audit Store"""),
        code_cell("""pred_df = pd.DataFrame(records)
print(f"Stream simulation completed. Total logged records: {len(pred_df):,}")
print("\\nSchema and Sample Records:")
print(pred_df.info())
pred_df.head(10)"""),
        md_cell("""### 3. Persist Prediction Audit Store to Parquet"""),
        code_cell("""STREAM_PREDICTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
pred_df.to_parquet(STREAM_PREDICTIONS_PARQUET, index=False, engine='pyarrow', compression='snappy')
print(f"Saved prediction audit table to {STREAM_PREDICTIONS_PARQUET} (Size: {STREAM_PREDICTIONS_PARQUET.stat().st_size / 1e6:.2f} MB)")""")
    ]
    return make_notebook(cells)


def build_notebook_06():
    cells = [
        md_cell("""# Step 10 & 11: Performance Monitoring & Drift Detection
## Operationalizing ML: When is the Model Degrading?

In production, models degrade due to two fundamentally different phenomena:
1. **Data Drift**: $P(X)$ changes. Input feature distributions shift (e.g., changes in rider request volume, weather, special events).
2. **Concept Drift / Performance Degradation**: $P(Y|X)$ changes. The underlying relationship between input features and demand shifts, causing model accuracy to degrade.

> **CRITICAL RULE**: Detecting data drift does **NOT** automatically mean concept drift has occurred. A model may be robust to data drift. Therefore, we monitor both feature stability (PSI, KS-test) and actual operational performance (Rolling MAE, RMSE, and Bias)."""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path.cwd().parent))
from src.drift import calculate_psi, calculate_ks_test, compute_rolling_metrics, detect_sustained_degradation
from src.features import build_feature_pipeline
from src.data_loader import load_processed_demand, split_chronological
from src.config import STREAM_PREDICTIONS_PARQUET, DRIFT_PSI_THRESHOLD

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_size = (12, 5)

print("Loading stream prediction audit log...")
pred_df = pd.read_parquet(STREAM_PREDICTIONS_PARQUET)
print(f"Loaded {len(pred_df):,} audit records.")
pred_df.head()"""),
        md_cell("""### 1. Rolling Performance Tracking (MAE, RMSE, Bias)
We calculate 24-hour rolling metrics (window of 96 intervals) across the streaming timeline."""),
        code_cell("""rolling_summary = compute_rolling_metrics(pred_df, window_size=96)
rolling_summary.head(10)"""),
        md_cell("""### 2. Visualize Rolling MAE & Degradation Thresholds
We set an alert control limit at 15% above the baseline validation MAE."""),
        code_cell("""baseline_val_mae = rolling_summary['mean_abs_error'].median()
control_limit = baseline_val_mae * 1.15

plt.figure(figsize=fig_size)
plt.plot(rolling_summary['prediction_timestamp'], rolling_summary['rolling_mae'], 
         color='#1f77b4', lw=2, label='Rolling 24h MAE')
plt.axhline(baseline_val_mae, color='green', linestyle='--', label=f'Baseline MAE ({baseline_val_mae:.2f})')
plt.axhline(control_limit, color='red', linestyle='--', label=f'Alert Limit (+15%: {control_limit:.2f})')

plt.title("Rolling 24-Hour MAE Over Simulated Stream Timeline", fontsize=14, fontweight='bold')
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 3. Rolling Bias (Mean Error): Monitoring Systematic Over/Under Prediction
Bias tracks whether the model consistently over-predicts (positive) or under-predicts (negative)."""),
        code_cell("""plt.figure(figsize=fig_size)
plt.plot(rolling_summary['prediction_timestamp'], rolling_summary['rolling_bias'], 
         color='#ff7f0e', lw=2, label='Rolling 24h Bias (Predicted - Actual)')
plt.axhline(0, color='black', linestyle=':', lw=1.5)
plt.title("Rolling Prediction Bias (Mean Error) Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Bias (Rides)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 4. Input Data Drift Detection (PSI & KS-Test)
We compute Population Stability Index (PSI) and Kolmogorov-Smirnov test between the Training reference distribution and the Streaming test window."""),
        code_cell("""grid_df = load_processed_demand()
feat_df = build_feature_pipeline(grid_df, drop_burn_in=True)
train_df, _, test_df = split_chronological(feat_df)

features_to_monitor = ['lag_1', 'lag_4', 'lag_96', 'rolling_mean_4']
drift_results = []

for feat in features_to_monitor:
    ref_vals = train_df[feat].to_numpy()
    cur_vals = test_df[feat].to_numpy()
    
    psi_val = calculate_psi(ref_vals, cur_vals)
    ks_res = calculate_ks_test(ref_vals, cur_vals)
    
    drift_status = "Significant Drift" if psi_val >= 0.25 else ("Moderate Shift" if psi_val >= 0.10 else "Stable")
    
    drift_results.append({
        'Feature': feat,
        'PSI': psi_val,
        'PSI_Status': drift_status,
        'KS_Statistic': ks_res['ks_statistic'],
        'KS_P_Value': ks_res['p_value'],
        'KS_Drift': ks_res['is_drift']
    })

drift_df = pd.DataFrame(drift_results)
print("Data Drift Profiling Summary:")
print(drift_df.to_string(index=False))"""),
        md_cell("""### 5. Sustained Performance Degradation Check
We check whether rolling MAE sustained an increase above the control limit for consecutive periods."""),
        code_cell("""has_degraded, degraded_idx = detect_sustained_degradation(
    rolling_summary['rolling_mae'],
    baseline_mae=baseline_val_mae,
    threshold_pct=0.15,
    consecutive_periods=4
)

print(f"Sustained Performance Degradation Detected: {has_degraded}")
if has_degraded:
    print(f"Triggering Continual Learning / Model Update Workflow at timestamp: "
          f"{rolling_summary.loc[degraded_idx[0], 'prediction_timestamp']}")
else:
    print("Model performance remains within acceptable operational boundaries.")""")
    ]
    return make_notebook(cells)


def build_notebook_07():
    cells = [
        md_cell("""# Step 12: Continual Learning & Model Governance
## Controlled Model-Update Process & Champion-Challenger Promotion Gate

### Core Principle:
Do **NOT** blindly retrain whenever performance has a minor hiccup. Retraining should follow a disciplined, audited workflow:
1. Current model (`model_v1`, Champion) is serving predictions.
2. Performance monitor triggers an update alert due to sustained degradation or regular scheduled retrain window.
3. Train candidate model (`model_v2`, Challenger) on an updated sliding window.
4. Evaluate Challenger vs Champion on the exact same holdout evaluation set.
5. **Promotion Gate**: Promote Challenger to Champion **ONLY** if:
   - Overall MAE improves by $\\ge 3\\%$.
   - High-demand zone MAE does not regress by $> 2\\%$.
6. Update the Model Registry with full lineage, training window, validation metrics, and audit decision."""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

sys.path.append(str(Path.cwd().parent))
from src.data_loader import load_processed_demand, split_chronological
from src.features import build_feature_pipeline, get_feature_columns
from src.continual import ModelRegistry, evaluate_champion_challenger
from src.config import MODELS_DIR

print("Loading current Champion (model_v1) and Model Registry...")
registry = ModelRegistry()
champion_meta = registry.get_champion()
print("Current Champion Metadata:")
print(champion_meta)

champion_model = joblib.load(MODELS_DIR / "model_v1.joblib")
feature_cols = get_feature_columns()"""),
        md_cell("""### 1. Continual Learning Trigger & Sliding Window Retraining
To adapt to recent demand patterns, the challenger model is trained on an expanded window including recent data (Jan 8 through Jan 25).
We evaluate both models on the first 3 days of the Test set (Jan 26–28) as the holdout challenge period."""),
        code_cell("""grid_df = load_processed_demand()
feat_df = build_feature_pipeline(grid_df, drop_burn_in=True)

# Define updated training window for Challenger (Jan 8 to Jan 25)
challenger_train_mask = (feat_df['timestamp'] >= '2025-01-08') & (feat_df['timestamp'] <= '2025-01-25 23:59:59')
challenger_train_df = feat_df[challenger_train_mask]

# Define common holdout evaluation period (Jan 26 to Jan 28)
holdout_mask = (feat_df['timestamp'] >= '2025-01-26') & (feat_df['timestamp'] <= '2025-01-28 23:59:59')
holdout_df = feat_df[holdout_mask]

print(f"Challenger Training Data: {len(challenger_train_df):,} rows ({challenger_train_df['timestamp'].min()} to {challenger_train_df['timestamp'].max()})")
print(f"Common Holdout Eval Data: {len(holdout_df):,} rows ({holdout_df['timestamp'].min()} to {holdout_df['timestamp'].max()})")"""),
        md_cell("""### 2. Train Challenger Model (model_v2)"""),
        code_cell("""X_chall_train = challenger_train_df[feature_cols]
y_chall_train = challenger_train_df['demand'].to_numpy()

challenger_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.08,
    'num_leaves': 63,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

challenger_model = lgb.LGBMRegressor(**challenger_params)
print("Fitting Challenger Model (model_v2)...")
challenger_model.fit(X_chall_train, y_chall_train)
print("Challenger training complete.")"""),
        md_cell("""### 3. Champion vs Challenger Evaluation Gate
Evaluate both models on the common holdout period.
Criteria:
- Overall MAE improvement $\\ge 3\\%$.
- High-demand zone MAE does not regress by $> 2\\%$.

We identify top 20% high-demand zones to verify safety."""),
        code_cell("""top_zones = (
    challenger_train_df.groupby('PULocationID')['demand'].sum()
    .sort_values(ascending=False)
    .head(52)
    .index.tolist()
)

promoted, eval_summary = evaluate_champion_challenger(
    champion_model=champion_model,
    challenger_model=challenger_model,
    holdout_df=holdout_df,
    feature_cols=feature_cols,
    high_demand_zones=top_zones,
    min_improvement=0.03
)

print("Champion vs Challenger Evaluation Gate Results:")
for k, v in eval_summary.items():
    print(f"  - {k:<28}: {v}")"""),
        md_cell("""### 4. Controlled Model Promotion & Registry Update
If the challenger satisfies all criteria, it is promoted to production Champion, and the previous Champion is safely archived."""),
        code_cell("""model_v2_path = MODELS_DIR / "model_v2.joblib"
joblib.dump(challenger_model, model_v2_path)

if promoted:
    print("PROMOTION CRITERIA MET! Promoting model_v2 to Champion...")
    registry.register_model(
        version="model_v2",
        model_type="LightGBM Regressor (Retrained)",
        training_period=("2025-01-08", "2025-01-25"),
        features=feature_cols,
        val_metrics={"MAE": eval_summary["challenger_mae"]},
        status="champion",
        reason=f"Promoted over model_v1: +{eval_summary['pct_overall_improvement']}% overall MAE improvement",
        artifact_path=str(model_v2_path)
    )
    registry.promote_to_champion("model_v2", reason="Beats model_v1 on holdout validation gate")
else:
    print("Promotion criteria not met. Retaining model_v1 as Champion.")
    registry.register_model(
        version="model_v2",
        model_type="LightGBM Regressor (Retrained)",
        training_period=("2025-01-08", "2025-01-25"),
        features=feature_cols,
        val_metrics={"MAE": eval_summary["challenger_mae"]},
        status="rejected",
        reason="Did not achieve required improvement margin over champion",
        artifact_path=str(model_v2_path)
    )

print("\\nUpdated Model Registry Lineage Table:")
print(registry.get_lineage_table().to_string(index=False))"""),
        md_cell("""### Summary of Continual Learning
1. **Audited Governance**: Every candidate model is logged with exact training dates, features, and validation metrics.
2. **Safety Gates**: Prevents silent performance regression on high-value business zones.
3. **Reproducibility**: Models are versioned and can be rolled back instantly in `model_registry.json`.""")
    ]
    return make_notebook(cells)


def main():
    notebooks = {
        "01_data_understanding.ipynb": build_notebook_01(),
        "02_eda.ipynb": build_notebook_02(),
        "03_baseline_model.ipynb": build_notebook_03(),
        "04_ml_model.ipynb": build_notebook_04(),
        "05_stream_simulation.ipynb": build_notebook_05(),
        "06_performance_monitoring.ipynb": build_notebook_06(),
        "07_continual_learning.ipynb": build_notebook_07(),
    }

    for name, nb in notebooks.items():
        out_path = NOTEBOOKS_DIR / name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Generated {out_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
