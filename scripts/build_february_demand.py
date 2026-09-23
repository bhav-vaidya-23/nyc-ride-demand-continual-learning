"""
Build the February 2025 15-minute demand dataset.
Maintains exact schema, 15-minute interval frequency, and 262 zone definitions as January.
Saves to data/processed/demand_15min_2025_02.parquet.
"""
import time
import pandas as pd
from src.config import (
    DEMAND_15MIN_PARQUET,
    DEMAND_15MIN_PARQUET_FEB,
    FEB_START_BOUND,
)
from src.data_loader import (
    load_raw_demand_columns,
    aggregate_to_15min_grid,
    save_processed_demand,
    load_processed_demand,
)


def main():
    print("=" * 80)
    print("BUILDING FEBRUARY 2025 15-MINUTE DEMAND DATASET")
    print("=" * 80)

    # Step 1: Load January zones for exact spatiotemporal consistency
    print("Step 1: Reading January zone definitions for consistency...")
    jan_df = load_processed_demand(str(DEMAND_15MIN_PARQUET))
    jan_zones = sorted(jan_df["PULocationID"].unique().tolist())
    print(f"Verified {len(jan_zones)} unique pickup zones in January baseline.")

    # Step 2: Load raw February trip records
    print("\nStep 2: Loading raw February trip records (request_datetime, PULocationID)...")
    t0 = time.time()
    raw_feb = load_raw_demand_columns(filter_dates=True, month="2025-02")
    load_time = time.time() - t0
    print(f"Loaded {len(raw_feb):,} filtered February records in {load_time:.2f}s.")

    # Step 3: Aggregate into regular spatiotemporal grid
    print("\nStep 3: Aggregating into 15-min spatiotemporal grid with zero-filling...")
    t1 = time.time()
    feb_grid = aggregate_to_15min_grid(
        raw_feb,
        fill_zeros=True,
        start_bound=FEB_START_BOUND,
        end_bound="2025-02-28 23:45:00",
        target_zones=jan_zones,
    )
    agg_time = time.time() - t1

    expected_intervals = 28 * 24 * 4  # 2,688 intervals in 28 days of Feb
    expected_rows = expected_intervals * len(jan_zones)  # 2,688 * 262 = 704,256
    print(f"Constructed grid of {len(feb_grid):,} rows in {agg_time:.2f}s.")
    print(f"Expected: {expected_rows:,} rows. Match: {len(feb_grid) == expected_rows}")
    print(f"Timestamp range: {feb_grid['timestamp'].min()} to {feb_grid['timestamp'].max()}")
    print(f"Unique zones: {feb_grid['PULocationID'].nunique()} (identical to January)")

    # Step 4: Save to Parquet
    print(f"\nStep 4: Saving to {DEMAND_15MIN_PARQUET_FEB}...")
    save_processed_demand(feb_grid, str(DEMAND_15MIN_PARQUET_FEB))
    file_size_mb = DEMAND_15MIN_PARQUET_FEB.stat().st_size / 1e6
    print(f"Saved successfully! File size: {file_size_mb:.2f} MB.")


if __name__ == "__main__":
    main()
