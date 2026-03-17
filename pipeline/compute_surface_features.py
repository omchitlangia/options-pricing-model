"""
compute_surface_features.py
----------------------------
Computes features used in volatility surface modeling from the cleaned
options dataset.

Features added:
  moneyness               = spot / strike
  log_moneyness           = log(spot / strike)
  sqrt_T                  = sqrt(time_to_maturity)
  moneyness_T_interaction = moneyness * time_to_maturity

Input:  data/processed/options_surface/options_dataset_clean.csv
Output: data/processed/options_surface/options_dataset_features.csv

Run from the project root:
    python -m pipeline.compute_surface_features
"""

import math
import pandas as pd

from config.constants import PROCESSED_SURFACE_CLEAN, PROCESSED_SURFACE_FEATURES
from utils.io import load_dataset, save_dataset


def compute_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    df["moneyness"] = df["spot"] / df["strike"]
    return df


def compute_log_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    df["log_moneyness"] = df.apply(
        lambda row: math.log(row["spot"] / row["strike"]), axis=1
    )
    return df


def compute_sqrt_T(df: pd.DataFrame) -> pd.DataFrame:
    df["sqrt_T"] = df["time_to_maturity"].apply(math.sqrt)
    return df


def compute_moneyness_T_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df["moneyness_T_interaction"] = df["moneyness"] * df["time_to_maturity"]
    return df


def print_feature_summary(df: pd.DataFrame) -> None:
    feature_cols = ["moneyness", "log_moneyness", "sqrt_T", "moneyness_T_interaction"]
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"\nFeature statistics:")
    print(df[feature_cols].describe().round(4).to_string())
    print("=" * 60)


def main():
    print("=" * 60)
    print("Feature Engineering — Surface Pipeline")
    print("=" * 60)

    df = load_dataset(PROCESSED_SURFACE_CLEAN)

    print("\nComputing features:")
    df = compute_moneyness(df)
    print("  moneyness               : spot / strike")
    df = compute_log_moneyness(df)
    print("  log_moneyness           : log(spot / strike)")
    df = compute_sqrt_T(df)
    print("  sqrt_T                  : sqrt(time_to_maturity)")
    df = compute_moneyness_T_interaction(df)
    print("  moneyness_T_interaction : moneyness * time_to_maturity")

    save_dataset(df, PROCESSED_SURFACE_FEATURES)
    print_feature_summary(df)


if __name__ == "__main__":
    main()
