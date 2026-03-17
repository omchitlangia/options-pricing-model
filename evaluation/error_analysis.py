"""
error_analysis.py
------------------
Reusable functions for segmenting and reporting pricing errors
by moneyness and maturity buckets.

Works with any DataFrame that has columns:
  moneyness, time_to_maturity, <error_col>

Does NOT load files or produce plots — purely analytical functions.
"""

import pandas as pd


def moneyness_bins(df: pd.DataFrame, col: str = "moneyness") -> pd.Series:
    """Categorise options into OTM / ATM / ITM moneyness buckets."""
    return pd.cut(
        df[col],
        bins=[0, 0.97, 1.03, float("inf")],
        labels=["OTM", "ATM", "ITM"],
        right=False,
    )


def maturity_bins(df: pd.DataFrame, col: str = "time_to_maturity") -> pd.Series:
    """Categorise options into adaptive Short / Medium / Long maturity buckets."""
    T_max = df[col].max()
    if T_max > 0.10:
        bins   = [0.0, 0.05, 0.10, T_max + 1e-6]
        labels = ["Short", "Medium", "Long"]
    else:
        bins   = [0.0, 0.05, T_max + 1e-6]
        labels = ["Short", "Medium"]
    return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)


def error_by_moneyness(df: pd.DataFrame, error_col: str) -> pd.Series:
    """Return mean error grouped by moneyness bucket."""
    return df.groupby(moneyness_bins(df), observed=True)[error_col].mean()


def error_by_maturity(df: pd.DataFrame, error_col: str) -> pd.Series:
    """Return mean error grouped by maturity bucket."""
    return df.groupby(maturity_bins(df), observed=True)[error_col].mean()


def print_error_report(df: pd.DataFrame, error_col: str, label: str = "") -> None:
    """Print a full error report: overall MAE + breakdowns by moneyness and maturity."""
    mae = df[error_col].abs().mean()
    prefix = f"[{label}] " if label else ""

    print(f"\n{prefix}Mean Absolute Error : {mae:.4f}")
    print(f"\n{prefix}Error by moneyness:")
    print(error_by_moneyness(df, error_col).to_string())
    print(f"\n{prefix}Error by maturity:")
    print(error_by_maturity(df, error_col).to_string())


def print_model_comparison(df: pd.DataFrame) -> None:
    """
    Print a side-by-side MAE comparison for all three models.
    Expects columns: bs_error, binomial_error, mc_error (all signed errors).
    """
    print("\nModel MAE Comparison")
    print("-" * 40)
    for col, name in [("bs_error", "Black-Scholes"), ("binomial_error", "Binomial"), ("mc_error", "Monte Carlo")]:
        if col in df.columns:
            print(f"  {name:<15} : {df[col].abs().mean():.4f}")

    print("\nBy Moneyness (mean signed error):")
    error_cols = [c for c in ["bs_error", "binomial_error", "mc_error"] if c in df.columns]
    bins = moneyness_bins(df)
    print(df.groupby(bins, observed=True)[error_cols].mean().to_string())

    print("\nBy Maturity (mean signed error):")
    bins = maturity_bins(df)
    print(df.groupby(bins, observed=True)[error_cols].mean().to_string())
