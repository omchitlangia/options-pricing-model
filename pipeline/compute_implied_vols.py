"""
compute_implied_vols.py
------------------------
Computes Black-Scholes implied volatility for every option in the
features dataset using the bisection solver in core/implied_vol.py.

For each row the solver finds σ* such that:
    BS_call(spot, strike, time_to_maturity, r, σ*) = mid

Rows where the solver fails (price below intrinsic value) are dropped.

Input:  data/processed/options_surface/options_dataset_features.csv
Output: data/processed/options_surface/options_surface_dataset.csv

Run from the project root:
    python -m pipeline.compute_implied_vols
"""

import math
import pandas as pd

from core.implied_vol import implied_vol_bisection
from config.constants import (
    PROCESSED_SURFACE_FEATURES,
    PROCESSED_SURFACE_DATASET,
    RISK_FREE_RATE,
)
from utils.io import load_dataset, save_dataset
from utils.logging import print_header


def solve_implied_vol(row: pd.Series, r: float):
    """Bisection IV solve for a single option row. Returns None on failure."""
    return implied_vol_bisection(
        market_price=row["mid"],
        S=row["spot"],
        K=row["strike"],
        T=row["time_to_maturity"],
        r=r,
    )


def compute_implied_vols(df: pd.DataFrame, r: float) -> pd.DataFrame:
    """
    Adds an implied_vol column to df by applying the bisection solver
    to every row. Rows where the solver returns None are dropped.
    """
    n_before = len(df)
    print(f"  Solving IV for {n_before:,} options using bisection...")

    df = df.copy()
    df["implied_vol"] = df.apply(lambda row: solve_implied_vol(row, r), axis=1)

    n_failed = df["implied_vol"].isna().sum()
    df = df.dropna(subset=["implied_vol"]).copy()

    print(f"  Solver failed on {n_failed} rows")
    print(f"  Rows after IV computation: {len(df):,}")
    return df


def print_iv_statistics(df: pd.DataFrame) -> None:
    iv = df["implied_vol"]
    print(f"\n  IV statistics:")
    print(f"    mean : {iv.mean():.4f}")
    print(f"    std  : {iv.std():.4f}")
    print(f"    min  : {iv.min():.4f}")
    print(f"    max  : {iv.max():.4f}")


def print_iv_by_moneyness(df: pd.DataFrame) -> None:
    bins   = [0.0, 0.97, 1.03, float("inf")]
    labels = ["OTM", "ATM", "ITM"]
    bucket = pd.cut(df["moneyness"], bins=bins, labels=labels, right=False)
    result = df.groupby(bucket, observed=True)["implied_vol"].mean()
    print(f"\n  Mean IV by moneyness bucket:")
    for b, v in result.items():
        print(f"    {b}: {v:.4f}")


def print_iv_by_maturity(df: pd.DataFrame) -> None:
    T_max  = df["time_to_maturity"].max()
    upper  = max(T_max + 1e-6, 3.0)
    bins   = [0.0, 1/12, 6/12, upper]
    labels = ["Short (<1m)", "Medium (1-6m)", "Long (>6m)"]
    bucket = pd.cut(
        df["time_to_maturity"], bins=bins, labels=labels, include_lowest=True
    )
    result = df.groupby(bucket, observed=True)["implied_vol"].mean()
    print(f"\n  Mean IV by maturity bucket:")
    for b, v in result.items():
        if not math.isnan(v):
            print(f"    {b}: {v:.4f}")


def print_validation_report(df: pd.DataFrame) -> None:
    print_header("DATASET VALIDATION REPORT")
    print(f"\nDataset size: {len(df):,} rows")
    print(f"\nRows per ticker:")
    for ticker, count in df.groupby("ticker").size().items():
        print(f"  {ticker}: {count}")
    print_iv_statistics(df)
    print_iv_by_moneyness(df)
    print_iv_by_maturity(df)
    print("=" * 60)


def main() -> None:
    print_header("Implied Volatility Computation — Surface Pipeline")
    print(f"Risk-free rate : {RISK_FREE_RATE}")

    df = load_dataset(PROCESSED_SURFACE_FEATURES)
    df = compute_implied_vols(df, RISK_FREE_RATE)
    save_dataset(df, PROCESSED_SURFACE_DATASET)
    print_validation_report(df)


if __name__ == "__main__":
    main()
