"""
experiments/step6_binomial/run.py
-----------------------------------
Step 6 — Binomial Tree Pricing (SPY)

Prices SPY options using the CRR binomial tree with per-option implied vol.
Compares against BS errors from Step 2.
Depends on: data/processed/spy_with_implied_vol.csv  (Step 4 output)

Run from the project root:
    python -m experiments.step6_binomial.run
"""

from utils.io import load_dataset, save_dataset
from utils.logging import print_header
from evaluation.evaluate_pricing_models import price_with_binomial, print_pricing_summary
from evaluation.error_analysis import print_error_report
from config.constants import PROCESSED_SPY_IV, PROCESSED_SPY_BINOMIAL


def main() -> None:
    print_header("Step 6 — Binomial Tree Pricing")

    df = load_dataset(PROCESSED_SPY_IV)
    df = price_with_binomial(df)
    save_dataset(df, PROCESSED_SPY_BINOMIAL)

    print_pricing_summary(df)
    print_error_report(df, error_col="binomial_error", label="Binomial")


if __name__ == "__main__":
    main()
