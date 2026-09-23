"""
Configuration parameters for the Real-Time Ride Demand Prediction project.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories and file paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Parquet file paths
RAW_PARQUET_FILE = RAW_DATA_DIR / "fhvhv_tripdata_2025-01.parquet"
RAW_PARQUET_FILE_FEB = RAW_DATA_DIR / "fhvhv_tripdata_2025-02.parquet"
# Fallback to notebooks directory if raw file is located there
NOTEBOOKS_PARQUET_FILE = PROJECT_ROOT / "notebooks" / "fhvhv_tripdata_2025-01.parquet"

DEMAND_15MIN_PARQUET = PROCESSED_DATA_DIR / "demand_15min_2025_01.parquet"
DEMAND_15MIN_PARQUET_FEB = PROCESSED_DATA_DIR / "demand_15min_2025_02.parquet"
STREAM_PREDICTIONS_PARQUET = PROCESSED_DATA_DIR / "stream_predictions.parquet"
MODEL_REGISTRY_JSON = MODELS_DIR / "model_registry.json"

# Reports and visualization directories
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Strict date boundaries
DATA_START_BOUND = "2025-01-01 00:00:00"
DATA_END_BOUND = "2025-02-01 00:00:00"

FEB_START_BOUND = "2025-02-01 00:00:00"
FEB_END_BOUND = "2025-03-01 00:00:00"

# Time aggregation parameters
FREQ = "15min"
PERIODS_PER_HOUR = 4
PERIODS_PER_DAY = 96
PERIODS_PER_WEEK = 672

# Chronological split definitions for January (Training and Initial Validation)
TRAIN_START = "2025-01-01 00:00:00"
TRAIN_END = "2025-01-20 23:59:59"

VAL_START = "2025-01-21 00:00:00"
VAL_END = "2025-01-25 23:59:59"

TEST_START = "2025-01-26 00:00:00"
TEST_END = "2025-01-31 23:59:59"

# Feature engineering parameters
LAG_STEPS = [1, 2, 4, 8, 96, 672]  # 15m, 30m, 1h, 2h, 1day, 1week
ROLLING_WINDOWS = [4, 8]             # 1h, 2h rolling statistics

# Operational monitoring & drift configuration
ROLLING_WINDOW_24H = 96              # 96 15-min intervals = 24 hours
ROLLING_WINDOW_3D = 288              # 288 15-min intervals = 3 days

# Operational Trigger Policy:
# A 15% degradation threshold was selected as the operational trigger for this experiment.
# It is an empirical engineering policy threshold rather than an intrinsic ML constant.
DRIFT_PSI_THRESHOLD = 0.25           # Significant population drift
DRIFT_DEGRADATION_THRESHOLD = 0.15   # 15% sustained MAE increase above baseline
DEGRADATION_CONSECUTIVE_WINDOWS = 4  # Consecutive periods exceeding threshold to confirm sustained degradation
PROMOTION_IMPROVEMENT_THRESHOLD = 0.03  # Candidate must beat champion by >= 3% MAE on unseen holdout

