"""
Precompute the 15-minute demand grid for January 2025.
Constructs a regular spatiotemporal grid (2,976 intervals * 262 zones = 779,712 rows)
and saves to data/processed/demand_15min_2025_01.parquet.
"""
import time
from src.data_loader import (
    load_raw_demand_columns,
    aggregate_to_15min_grid,
    save_processed_demand,
)
from src.config import DEMAND_15MIN_PARQUET


def main():
    print("Step 1: Loading raw demand columns (request_datetime, PULocationID)...")
    t0 = time.time()
    df = load_raw_demand_columns(filter_dates=True)
    load_time = time.time() - t0
    print(f"Loaded {len(df):,} filtered records in {load_time:.2f}s.")

    print("\nStep 2: Aggregating into 15-minute spatiotemporal grid with zero-filling...")
    t1 = time.time()
    grid_df = aggregate_to_15min_grid(df, fill_zeros=True)
    agg_time = time.time() - t1
    print(f"Constructed grid of {len(grid_df):,} rows in {agg_time:.2f}s.")
    print("Grid summary:")
    print(grid_df.info())
    print("\nSample rows:")
    print(grid_df.head(10))

    print(f"\nStep 3: Saving to {DEMAND_15MIN_PARQUET}...")
    save_path = save_processed_demand(grid_df)
    print(f"Saved successfully to {save_path} (Size: {DEMAND_15MIN_PARQUET.stat().st_size / 1e6:.2f} MB).")


if __name__ == "__main__":
    main()
