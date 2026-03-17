"""
experiments/step4_iv/run.py
-----------------------------
Step 4 — Implied Volatility Extraction (SPY)

Extracts implied volatility from SPY market prices using the bisection solver.
Depends on: data/processed/spy_priced_bs.csv  (Step 2 output)

Run from the project root:
    python -m experiments.step4_iv.run
"""

from utils.io import load_dataset, save_dataset
from utils.logging import print_header
from evaluation.evaluate_iv_models import compute_implied_vols, print_iv_summary
from evaluation.error_analysis import moneyness_bins, maturity_bins
from config.constants import PROCESSED_SPY_BS, PROCESSED_SPY_IV
import pandas as pd


def print_iv_structure(df: pd.DataFrame) -> None:
    print("\nIV by moneyness (call smile):")
    print(df.groupby(moneyness_bins(df), observed=True)["implied_vol"].mean().to_string())
    print("\nIV by maturity (term structure):")
    print(df.groupby(maturity_bins(df), observed=True)["implied_vol"].mean().to_string())


def main() -> None:
    print_header("Step 4 — Implied Volatility Extraction")

    df = load_dataset(PROCESSED_SPY_BS)
    df = compute_implied_vols(df)
    save_dataset(df, PROCESSED_SPY_IV)

    print_iv_summary(df)
    print_iv_structure(df)


if __name__ == "__main__":
    main()
