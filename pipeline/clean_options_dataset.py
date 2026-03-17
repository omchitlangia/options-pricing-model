"""
clean_options_dataset.py
------------------------
Cleans the raw multi-asset options dataset collected by
pipeline/collect_options_surface.py.

Cleaning steps:
  1. Remove rows with invalid or missing prices (bid, ask, mid <= 0).
  2. Remove rows with extremely wide bid-ask spreads.
  3. Convert expiry column to datetime.
  4. Compute time_to_maturity in years (from today to expiry date).
  5. Remove rows with non-positive time to maturity (expired contracts).

Input:  data/raw/options_large/options_dataset_raw.csv
Output: data/processed/options_surface/options_dataset_clean.csv

Run from the project root:
    python -m pipeline.clean_options_dataset
"""

import pandas as pd
from datetime import datetime, timezone

from config.constants import (
    RAW_OPTIONS_SURFACE,
    PROCESSED_SURFACE_CLEAN,
    SURFACE_MAX_SPREAD_RATIO,
)
from utils.io import load_dataset, save_dataset

MAX_SPREAD_RATIO = SURFACE_MAX_SPREAD_RATIO


def remove_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows where bid, ask, or mid are non-positive or missing."""
    n_before = len(df)
    df = df.dropna(subset=["bid", "ask", "mid"])
    df = df[
        (df["bid"] >= 0) &
        (df["ask"] > 0) &
        (df["mid"] > 0) &
        (df["ask"] > df["bid"])
    ].copy()
    print(f"  remove_invalid_prices: {n_before} → {len(df)} rows")
    return df


def remove_wide_spreads(df: pd.DataFrame, max_spread_ratio: float) -> pd.DataFrame:
    """Drops rows where the bid-ask spread is disproportionately wide."""
    n_before = len(df)
    spread_ratio = (df["ask"] - df["bid"]) / df["mid"]
    df = df[spread_ratio <= max_spread_ratio].copy()
    print(f"  remove_wide_spreads (>{max_spread_ratio*100:.0f}% of mid): {n_before} → {len(df)} rows")
    return df


def convert_expiry_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the expiry column from string to datetime.date."""
    n_before = len(df)
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    df = df.dropna(subset=["expiry"]).copy()
    print(f"  convert_expiry_to_datetime: {n_before} → {len(df)} rows")
    return df


def compute_time_to_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """Computes time_to_maturity in years as (expiry_date - today) / 365.0."""
    today = datetime.now(timezone.utc).date()
    df["time_to_maturity"] = df["expiry"].apply(
        lambda exp: (exp - today).days / 365.0
    )
    print(f"  compute_time_to_maturity: reference date = {today}")
    return df


def remove_expired(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows where time_to_maturity is zero or negative."""
    n_before = len(df)
    df = df[df["time_to_maturity"] > 0].copy()
    print(f"  remove_expired: {n_before} → {len(df)} rows")
    return df


def print_clean_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("CLEAN DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"\nRows per ticker:")
    for ticker, count in df.groupby("ticker").size().items():
        print(f"  {ticker}: {count}")
    print(f"\ntime_to_maturity range: {df['time_to_maturity'].min():.4f} — {df['time_to_maturity'].max():.4f} years")
    print(f"strike range: {df['strike'].min():.2f} — {df['strike'].max():.2f}")
    print(f"mid price range: {df['mid'].min():.4f} — {df['mid'].max():.2f}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Data Cleaning — Surface Pipeline")
    print("=" * 60)

    df = load_dataset(RAW_OPTIONS_SURFACE)

    print("\nApplying filters:")
    df = remove_invalid_prices(df)
    df = remove_wide_spreads(df, MAX_SPREAD_RATIO)
    df = convert_expiry_to_datetime(df)
    df = compute_time_to_maturity(df)
    df = remove_expired(df)

    save_dataset(df, PROCESSED_SURFACE_CLEAN)
    print_clean_summary(df)


if __name__ == "__main__":
    main()
