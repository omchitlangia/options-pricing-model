"""
experiments/step2_black_scholes/run.py
----------------------------------------
Step 2 — Black-Scholes Pricing on SPY

Prices SPY options using Black-Scholes with constant historical volatility.
Saves the priced dataset and prints a full error report.

Run from the project root:
    python -m experiments.step2_black_scholes.run
"""

from utils.io import load_dataset, save_dataset
from utils.logging import print_header
from evaluation.evaluate_pricing_models import price_with_bs, print_pricing_summary
from evaluation.error_analysis import print_error_report
from config.constants import PROCESSED_SPY_CLEAN, PROCESSED_SPY_BS


def main() -> None:
    print_header("Step 2 — Black-Scholes Pricing")

    df = load_dataset(PROCESSED_SPY_CLEAN)
    df = price_with_bs(df)
    save_dataset(df, PROCESSED_SPY_BS)

    print_pricing_summary(df)
    print_error_report(df, error_col="bs_error", label="BS")


if __name__ == "__main__":
    main()
