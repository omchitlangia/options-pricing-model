"""
experiments/step7_monte_carlo/run.py
--------------------------------------
Step 7 — Monte Carlo Pricing (SPY)

Prices SPY options using Monte Carlo GBM simulation with per-option implied vol.
Depends on: data/processed/spy_with_implied_vol.csv  (Step 4 output)

Run from the project root:
    python -m experiments.step7_monte_carlo.run
"""

from utils.io import load_dataset, save_dataset
from utils.logging import print_header
from evaluation.evaluate_pricing_models import price_with_mc, print_pricing_summary
from evaluation.error_analysis import print_error_report
from config.constants import PROCESSED_SPY_IV, PROCESSED_SPY_MC


def main() -> None:
    print_header("Step 7 — Monte Carlo Pricing")

    df = load_dataset(PROCESSED_SPY_IV)
    df = price_with_mc(df)
    save_dataset(df, PROCESSED_SPY_MC)

    print_pricing_summary(df)
    print_error_report(df, error_col="mc_error", label="MC")


if __name__ == "__main__":
    main()
