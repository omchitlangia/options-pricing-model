"""
experiments/step3_real_data/run.py
------------------------------------
Step 3 — Data Collection and Cleaning (SPY)

Fetches raw SPY options from yfinance and applies the cleaning pipeline.
This step produces the clean SPY dataset used by all subsequent steps.

Run from the project root:
    python -m experiments.step3_real_data.run
"""

from utils.logging import print_header
from pipeline.pull_spy_options import fetch_spy_options
from pipeline.clean_spy_options import clean_spy_options
from utils.io import save_dataset
from config.constants import RAW_SPY_OPTIONS, PROCESSED_SPY_CLEAN


def main() -> None:
    print_header("Step 3 — Data Collection and Cleaning")

    print("\n[1/2] Fetching raw SPY option chains...")
    df_raw = fetch_spy_options()
    save_dataset(df_raw, RAW_SPY_OPTIONS)

    print("\n[2/2] Applying cleaning pipeline...")
    df_clean = clean_spy_options(df_raw)
    save_dataset(df_clean, PROCESSED_SPY_CLEAN)

    print(f"\nRows raw    : {len(df_raw)}")
    print(f"Rows clean  : {len(df_clean)}")


if __name__ == "__main__":
    main()
