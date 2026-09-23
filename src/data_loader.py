"""
Data loading, inspection, boundary filtering, and aggregation utilities.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pyarrow.parquet as pq
from src.config import (
    RAW_PARQUET_FILE,
    RAW_PARQUET_FILE_FEB,
    NOTEBOOKS_PARQUET_FILE,
    DEMAND_15MIN_PARQUET,
    DEMAND_15MIN_PARQUET_FEB,
    DATA_START_BOUND,
    DATA_END_BOUND,
    FEB_START_BOUND,
    FEB_END_BOUND,
    FREQ,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
)


def get_raw_parquet_path(month: str = "2025-01") -> str:
    """Resolve and return the valid path to the raw Parquet file."""
    if month == "2025-02":
        if RAW_PARQUET_FILE_FEB.exists():
            return str(RAW_PARQUET_FILE_FEB)
        raise FileNotFoundError(f"February raw parquet file not found in {RAW_PARQUET_FILE_FEB}")
    
    if RAW_PARQUET_FILE.exists():
        return str(RAW_PARQUET_FILE)
    if NOTEBOOKS_PARQUET_FILE.exists():
        return str(NOTEBOOKS_PARQUET_FILE)
    raise FileNotFoundError(
        f"Raw parquet file not found in {RAW_PARQUET_FILE} or {NOTEBOOKS_PARQUET_FILE}"
    )


def inspect_raw_metadata(month: str = "2025-01") -> Dict:
    """
    Inspect Parquet file metadata efficiently without loading all rows into RAM.
    """
    path = get_raw_parquet_path(month=month)
    pf = pq.ParquetFile(path)
    metadata = pf.metadata
    schema = pf.schema_arrow

    return {
        "file_path": path,
        "num_rows": metadata.num_rows,
        "num_row_groups": pf.num_row_groups,
        "num_columns": len(schema.names),
        "column_names": schema.names,
        "schema": {name: str(schema.field(name).type) for name in schema.names},
    }


def load_raw_demand_columns(
    columns: Optional[List[str]] = None,
    filter_dates: bool = True,
    month: str = "2025-01",
) -> pd.DataFrame:
    """
    Load raw trip records, selecting only specified columns to maintain memory efficiency.

    Args:
        columns: List of columns to load. Defaults to ['request_datetime', 'PULocationID'].
        filter_dates: If True, filters strictly to month boundary.
        month: "2025-01" or "2025-02".

    Returns:
        pd.DataFrame containing filtered records.
    """
    if columns is None:
        columns = ["request_datetime", "PULocationID"]

    path = get_raw_parquet_path(month=month)
    df = pd.read_parquet(path, columns=columns)

    if filter_dates and "request_datetime" in df.columns:
        start_bound = FEB_START_BOUND if month == "2025-02" else DATA_START_BOUND
        end_bound = FEB_END_BOUND if month == "2025-02" else DATA_END_BOUND
        mask = (df["request_datetime"] >= start_bound) & (
            df["request_datetime"] < end_bound
        )
        df = df.loc[mask].reset_index(drop=True)

    return df


def aggregate_to_15min_grid(
    df: pd.DataFrame,
    fill_zeros: bool = True,
    start_bound: Optional[str] = None,
    end_bound: Optional[str] = None,
    target_zones: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Aggregate trip requests into 15-minute intervals per pickup zone.
    Constructs a regular spatiotemporal grid with complete Cartesian product
    and fills intervals with 0 demand.

    Args:
        df: Raw DataFrame with 'request_datetime' and 'PULocationID'.
        fill_zeros: If True, builds a complete (timestamp x PULocationID) grid.
        start_bound: Start timestamp for grid (default: DATA_START_BOUND).
        end_bound: End timestamp for grid (default: '2025-01-31 23:45:00').
        target_zones: Fixed list of zones (e.g. 262 January zones) for exact schema consistency.

    Returns:
        pd.DataFrame with columns ['timestamp', 'PULocationID', 'demand']
        sorted by ['PULocationID', 'timestamp'].
    """
    df = df.copy()
    if target_zones is not None:
        df = df[df["PULocationID"].isin(target_zones)].copy()

    # Create 15-minute bucket
    df["timestamp"] = df["request_datetime"].dt.floor(FREQ)

    # Group and aggregate demand
    agg = (
        df.groupby(["timestamp", "PULocationID"], as_index=False)
        .size()
        .rename(columns={"size": "demand"})
    )

    if fill_zeros:
        grid_start = start_bound or DATA_START_BOUND
        grid_end = end_bound or "2025-01-31 23:45:00"
        full_timestamps = pd.date_range(
            start=grid_start,
            end=grid_end,
            freq=FREQ,
            name="timestamp",
        )
        all_zones = sorted(target_zones if target_zones is not None else df["PULocationID"].unique())

        # Cartesian product index: intervals * zones
        full_index = pd.MultiIndex.from_product(
            [full_timestamps, all_zones], names=["timestamp", "PULocationID"]
        )

        agg = (
            agg.set_index(["timestamp", "PULocationID"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

    # Ensure memory-friendly types
    agg["PULocationID"] = agg["PULocationID"].astype("int32")
    agg["demand"] = agg["demand"].astype("int32")

    # Sort stably by zone then time for lag computations
    agg = agg.sort_values(by=["PULocationID", "timestamp"]).reset_index(drop=True)
    return agg


def save_processed_demand(
    df: pd.DataFrame, output_path: Optional[str] = None
) -> str:
    """Save processed 15-min demand dataset to Parquet."""
    out = output_path or str(DEMAND_15MIN_PARQUET)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    return out





def load_processed_demand(input_path: Optional[str] = None) -> pd.DataFrame:
    """Load processed 15-min demand dataset from Parquet."""
    path = input_path or str(DEMAND_15MIN_PARQUET)
    return pd.read_parquet(path)


def split_chronological(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically into Train, Validation, and Test sets.
    Prevents any temporal lookahead leakage.

    Returns:
        (train_df, val_df, test_df)
    """
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    val_df = df.loc[val_mask].copy().reset_index(drop=True)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    return train_df, val_df, test_df
