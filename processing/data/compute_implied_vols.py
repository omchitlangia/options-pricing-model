"""
compute_implied_vols.py
------------------------
Computes Black–Scholes implied volatility for every option in the
features dataset using the bisection solver from volatility/implied_vol.py —
the same solver used in the existing single-asset pipeline.

For each row, the solver finds σ* such that:
    BS_call(spot, strike, time_to_maturity, r, σ*) = mid

Rows where the solver fails (e.g. price below intrinsic value) are dropped.

After solving, prints a full dataset validation report:
  - total rows, rows per ticker
  - IV statistics (mean, std, min, max)
  - mean IV by moneyness bucket (OTM / ATM / ITM)
  - mean IV by maturity bucket (Short / Medium / Long)

Input:  data/processed/options_surface/options_dataset_features.csv
Output: data/processed/options_surface/options_surface_dataset.csv

Run from the project root:
    python processing/compute_implied_vols.py
"""

import math
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path so the volatility package is importable
# regardless of which directory this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from volatility.implied_vol import implied_vol_bisection


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

INPUT_PATH  = "data/processed/options_surface/options_dataset_features.csv"
OUTPUT_PATH = "data/processed/options_surface/options_surface_dataset.csv"

RISK_FREE_RATE = 0.036   # consistent with the rest of the project


# -----------------------------------------------------------------------
# Implied volatility computation
# -----------------------------------------------------------------------

def solve_implied_vol(row: pd.Series, r: float):
    """
    Calls the bisection IV solver for a single option row.
    Returns None if the solver cannot find a valid solution.
    """
    return implied_vol_bisection(
        market_price=row["mid"],
        S=row["spot"],
        K=row["strike"],
        T=row["time_to_maturity"],
        r=r
    )


def compute_implied_vols(df: pd.DataFrame, r: float) -> pd.DataFrame:
    """
    Adds an implied_vol column to df by applying the bisection solver
    to every row. Rows where the solver returns None are dropped.
    """
    n_before = len(df)
    print(f"  Solving IV for {n_before} options using bisection...")

    df = df.copy()
    df["implied_vol"] = df.apply(lambda row: solve_implied_vol(row, r), axis=1)

    n_failed = df["implied_vol"].isna().sum()
    df = df.dropna(subset=["implied_vol"]).copy()

    print(f"  Solver failed on {n_failed} rows (price below intrinsic or numerical issues)")
    print(f"  Rows after IV computation: {len(df)}")

    return df


# -----------------------------------------------------------------------
# Validation and reporting
# -----------------------------------------------------------------------

def print_iv_statistics(df: pd.DataFrame) -> None:
    """Prints descriptive statistics for implied_vol."""
    iv = df["implied_vol"]
    print(f"\n  IV statistics:")
    print(f"    mean : {iv.mean():.4f}")
    print(f"    std  : {iv.std():.4f}")
    print(f"    min  : {iv.min():.4f}")
    print(f"    max  : {iv.max():.4f}")


def print_iv_by_moneyness(df: pd.DataFrame) -> None:
    """
    Prints mean implied volatility segmented by moneyness bucket.

    OTM : moneyness < 0.97   (spot well below strike)
    ATM : 0.97 ≤ moneyness ≤ 1.03
    ITM : moneyness > 1.03   (spot well above strike)
    """
    bins   = [0.0, 0.97, 1.03, float("inf")]
    labels = ["OTM", "ATM", "ITM"]
    moneyness_bucket = pd.cut(df["moneyness"], bins=bins, labels=labels, right=False)
    result = df.groupby(moneyness_bucket, observed=True)["implied_vol"].mean()
    print(f"\n  Mean IV by moneyness bucket:")
    for bucket, val in result.items():
        print(f"    {bucket}: {val:.4f}")


def print_iv_by_maturity(df: pd.DataFrame) -> None:
    """
    Prints mean implied volatility segmented by time to maturity.

    Short  : T < 1/12 year  (< 1 month)
    Medium : 1/12 ≤ T < 6/12  (1–6 months)
    Long   : T ≥ 6/12  (6+ months)
    """
    T_max = df["time_to_maturity"].max()
    upper = max(T_max + 1e-6, 3.0)   # ensure the last bin is never empty

    bins   = [0.0, 1/12, 6/12, upper]
    labels = ["Short (<1m)", "Medium (1–6m)", "Long (>6m)"]

    maturity_bucket = pd.cut(
        df["time_to_maturity"], bins=bins, labels=labels, include_lowest=True
    )
    result = df.groupby(maturity_bucket, observed=True)["implied_vol"].mean()
    print(f"\n  Mean IV by maturity bucket:")
    for bucket, val in result.items():
        if not math.isnan(val):
            print(f"    {bucket}: {val:.4f}")


def print_validation_report(df: pd.DataFrame) -> None:
    """Prints the full dataset validation report."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"\nDataset size: {len(df)} rows")

    print(f"\nRows per ticker:")
    for ticker, count in df.groupby("ticker").size().items():
        print(f"  {ticker}: {count}")

    print_iv_statistics(df)
    print_iv_by_moneyness(df)
    print_iv_by_maturity(df)

    print("=" * 60)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 4 — Implied Volatility Computation")
    print("=" * 60)
    print(f"Risk-free rate: {RISK_FREE_RATE}")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    print("\nComputing implied volatilities:")
    df = compute_implied_vols(df, RISK_FREE_RATE)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFinal dataset saved to: {OUTPUT_PATH}")

    print_validation_report(df)


if __name__ == "__main__":
    main()
