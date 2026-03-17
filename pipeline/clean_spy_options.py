"""
clean_spy_options.py
---------------------
Cleans the raw SPY options dataset: computes mid price, features,
applies liquidity and moneyness filters, and saves the clean output.

Standardised output schema:
  spot, strike, mid, time_to_maturity, moneyness, bid, ask, dte, ...

Input:  data/raw/spy_options_raw.csv
Output: data/processed/spy_options_clean.csv

Run from the project root:
    python -m pipeline.clean_spy_options
"""

import pandas as pd

from config.constants import (
    RAW_SPY_OPTIONS,
    PROCESSED_SPY_CLEAN,
    SPY_MONEYNESS_LOW,
    SPY_MONEYNESS_HIGH,
)
from utils.io import load_dataset, save_dataset


def clean_spy_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mid price, moneyness, time_to_maturity and apply
    liquidity / moneyness filters. Returns the cleaned DataFrame.
    """
    df = df.copy()

    df["mid"]              = (df["bid"] + df["ask"]) / 2
    df["moneyness"]        = df["spot"] / df["strike"]
    df["time_to_maturity"] = df["dte"] / 365.0

    df_clean = df[
        (df["bid"] > 0) &
        (df["ask"] > df["bid"]) &
        ((df["volume"] >= 10) | (df["openInterest"] >= 100)) &
        (df["moneyness"].between(SPY_MONEYNESS_LOW, SPY_MONEYNESS_HIGH))
    ].copy()

    return df_clean


def main() -> None:
    df_raw   = load_dataset(RAW_SPY_OPTIONS)
    df_clean = clean_spy_options(df_raw)
    save_dataset(df_clean, PROCESSED_SPY_CLEAN)
    print(f"Rows before cleaning : {len(df_raw)}")
    print(f"Rows after  cleaning : {len(df_clean)}")


if __name__ == "__main__":
    main()
