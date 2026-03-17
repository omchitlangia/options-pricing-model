"""
pull_spy_options.py
--------------------
Fetches raw SPY option chains from yfinance and saves to CSV.

Output: data/raw/spy_options_raw.csv

Run from the project root:
    python -m pipeline.pull_spy_options
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

from config.constants import RAW_SPY_OPTIONS, SPY_MIN_DTE, SPY_MAX_DTE
from utils.io import save_dataset


def fetch_spy_options(
    ticker_symbol: str = "SPY",
    min_dte: int = SPY_MIN_DTE,
    max_dte: int = SPY_MAX_DTE,
) -> pd.DataFrame:
    """
    Fetch all SPY call chains within the given DTE window.
    Returns a raw DataFrame with spot price and expiry metadata attached.
    """
    ticker = yf.Ticker(ticker_symbol)
    spot   = ticker.history(period="1d")["Close"].iloc[-1]
    today  = datetime.now(timezone.utc).date()

    rows = []
    for exp in ticker.options:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        if min_dte <= dte <= max_dte:
            calls = ticker.option_chain(exp).calls.copy()
            calls["expiry"] = exp_date
            calls["dte"]    = dte
            calls["spot"]   = spot
            rows.append(calls)

    df = pd.concat(rows, ignore_index=True)
    print(f"Spot price : {spot:.2f}")
    print(f"Expiries   : {len(rows)}")
    return df


def main() -> None:
    df = fetch_spy_options()
    save_dataset(df, RAW_SPY_OPTIONS)


if __name__ == "__main__":
    main()
