"""
experiments/step5_greeks/run.py
---------------------------------
Step 5 — Greeks Computation (SPY)

Computes analytical Black-Scholes Greeks using per-option implied volatility.
Depends on: data/processed/spy_with_implied_vol.csv  (Step 4 output)

Run from the project root:
    python -m experiments.step5_greeks.run
"""

from utils.io import load_dataset, save_dataset
from utils.logging import print_header
from evaluation.evaluate_iv_models import compute_greeks, print_greeks_summary
from config.constants import PROCESSED_SPY_IV, PROCESSED_SPY_GREEKS


def main() -> None:
    print_header("Step 5 — Greeks Computation")

    df = load_dataset(PROCESSED_SPY_IV)
    df = compute_greeks(df)
    save_dataset(df, PROCESSED_SPY_GREEKS)

    print_greeks_summary(df)


if __name__ == "__main__":
    main()
