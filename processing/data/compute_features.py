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

These features characterize each option's position in the (strike, maturity)
space and are standard inputs to smile interpolation, regression models,
and machine learning approaches.

Input:  data/processed/options_surface/options_dataset_clean.csv
Output: data/processed/options_surface/options_dataset_features.csv

Run from the project root:
    python -m processing.data.compute_features
"""

import math
import pandas as pd


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

INPUT_PATH  = "data/processed/options_surface/options_dataset_clean.csv"
OUTPUT_PATH = "data/processed/options_surface/options_dataset_features.csv"


# -----------------------------------------------------------------------
# Feature engineering functions
# -----------------------------------------------------------------------

def compute_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moneyness = spot / strike.

    For a call option:
      moneyness > 1 → in-the-money (spot above strike)
      moneyness ≈ 1 → at-the-money
      moneyness < 1 → out-of-the-money (spot below strike)

    This is the same definition used throughout the existing project
    (see scripts/clean_spy_options.py).
    """
    df["moneyness"] = df["spot"] / df["strike"]
    return df


def compute_log_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-moneyness = log(spot / strike).

    The log-moneyness is symmetric around zero (zero = at-the-money),
    which makes it a natural coordinate for smile modeling. It appears
    in the Black–Scholes d1/d2 terms and in the SVI parametrization.
    """
    df["log_moneyness"] = df.apply(
        lambda row: math.log(row["spot"] / row["strike"]),
        axis=1
    )
    return df


def compute_sqrt_T(df: pd.DataFrame) -> pd.DataFrame:
    """
    sqrt_T = sqrt(time_to_maturity).

    Implied volatility scales approximately with 1/sqrt(T) in many
    parametric smile models. Using sqrt_T as a feature instead of T
    linearizes this relationship and improves regression model fits.
    """
    df["sqrt_T"] = df["time_to_maturity"].apply(math.sqrt)
    return df


def compute_moneyness_T_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    moneyness_T_interaction = moneyness * time_to_maturity.

    Captures the joint effect of strike position and time horizon.
    At-the-money long-dated options behave differently from at-the-money
    short-dated options; this interaction term helps regression models
    learn that relationship.
    """
    df["moneyness_T_interaction"] = df["moneyness"] * df["time_to_maturity"]
    return df


def print_feature_summary(df: pd.DataFrame) -> None:
    """Prints descriptive statistics for all computed features."""
    feature_cols = ["moneyness", "log_moneyness", "sqrt_T", "moneyness_T_interaction"]
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"\nFeature statistics:")
    print(df[feature_cols].describe().round(4).to_string())
    print("=" * 60)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 3 — Feature Engineering")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    print("\nComputing features:")

    df = compute_moneyness(df)
    print("  moneyness               : spot / strike")

    df = compute_log_moneyness(df)
    print("  log_moneyness           : log(spot / strike)")

    df = compute_sqrt_T(df)
    print("  sqrt_T                  : sqrt(time_to_maturity)")

    df = compute_moneyness_T_interaction(df)
    print("  moneyness_T_interaction : moneyness * time_to_maturity")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeatures dataset saved to: {OUTPUT_PATH}")

    print_feature_summary(df)


if __name__ == "__main__":
    main()
