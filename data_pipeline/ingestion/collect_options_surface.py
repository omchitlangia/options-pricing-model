"""
collect_options_surface.py
--------------------------
Downloads call options chains for multiple liquid US equity underlyings
using yfinance. Collects up to 12 expiries per ticker, computes mid price,
and saves a raw dataset suitable for volatility surface modeling.

Output: data/raw/options_large/options_dataset_raw.csv

Run from the project root:
    python scripts/collect_options_surface.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]

MAX_EXPIRIES = 12       # collect at most 12 expiries per ticker

OUTPUT_PATH = "data/raw/options_large/options_dataset_raw.csv"


# -----------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------

def fetch_spot_price(ticker_obj: yf.Ticker, ticker: str) -> float:
    """
    Returns the most recent closing price for the underlying.
    """
    hist = ticker_obj.history(period="1d")
    if hist.empty:
        raise ValueError(f"No price history available for {ticker}")
    return float(hist["Close"].iloc[-1])


def fetch_calls_for_expiry(
    ticker_obj: yf.Ticker,
    expiry: str,
    spot: float,
    ticker: str
) -> pd.DataFrame:
    """
    Fetches the call option chain for a single expiry date.

    Extracts the required fields and attaches ticker, expiry, and spot.
    Computes mid price as (bid + ask) / 2.
    Returns an empty DataFrame if the chain cannot be fetched.
    """
    try:
        chain = ticker_obj.option_chain(expiry).calls.copy()
    except Exception as e:
        print(f"    Warning: could not fetch {ticker} expiry {expiry} — {e}")
        return pd.DataFrame()

    if chain.empty:
        return pd.DataFrame()

    # Select and rename the required fields
    required_cols = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]
    missing = [c for c in required_cols if c not in chain.columns]
    if missing:
        print(f"    Warning: missing columns {missing} for {ticker} {expiry}, skipping")
        return pd.DataFrame()

    row = chain[required_cols].copy()
    row["ticker"] = ticker
    row["expiry"] = expiry
    row["spot"] = spot

    # Compute mid price
    row["mid"] = (row["bid"] + row["ask"]) / 2

    # Remove options where mid is not positive — no reliable price
    row = row[row["mid"] > 0].copy()

    return row


def collect_ticker(ticker: str, max_expiries: int) -> pd.DataFrame:
    """
    Collects call options for all available expiries (up to max_expiries)
    for a single ticker. Returns a combined DataFrame.
    """
    print(f"\n[{ticker}] Fetching data...")

    ticker_obj = yf.Ticker(ticker)

    # Spot price
    try:
        spot = fetch_spot_price(ticker_obj, ticker)
    except ValueError as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

    print(f"  Spot price: {spot:.2f}")

    # Available expiries
    try:
        all_expiries = ticker_obj.options
    except Exception as e:
        print(f"  Error fetching expiry list: {e}")
        return pd.DataFrame()

    expiries = list(all_expiries)[:max_expiries]
    print(f"  Expiries to collect ({len(expiries)}): {expiries}")

    frames = []
    for exp in expiries:
        df_exp = fetch_calls_for_expiry(ticker_obj, exp, spot, ticker)
        if not df_exp.empty:
            frames.append(df_exp)
            print(f"    {exp}: {len(df_exp)} calls collected")

    if not frames:
        print(f"  No data collected for {ticker}")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def print_summary(df: pd.DataFrame) -> None:
    """
    Prints a collection summary to stdout.
    """
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total rows collected : {len(df)}")
    print(f"Tickers              : {sorted(df['ticker'].unique())}")

    print("\nRows per ticker:")
    for ticker, count in df.groupby("ticker").size().items():
        print(f"  {ticker}: {count}")

    print("\nExpiries collected per ticker:")
    for ticker in sorted(df["ticker"].unique()):
        exps = sorted(df[df["ticker"] == ticker]["expiry"].unique())
        print(f"  {ticker} ({len(exps)} expiries): {exps}")

    print("=" * 60)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    today = datetime.now(timezone.utc).date()
    print(f"Options surface data collection — {today}")
    print(f"Tickers: {TICKERS}")
    print(f"Max expiries per ticker: {MAX_EXPIRIES}")
    print(f"Output: {OUTPUT_PATH}")

    all_frames = []

    for ticker in TICKERS:
        df_ticker = collect_ticker(ticker, MAX_EXPIRIES)
        if not df_ticker.empty:
            all_frames.append(df_ticker)

    if not all_frames:
        print("\nNo data was collected. Check your internet connection or ticker symbols.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)

    # Reorder columns for clarity
    col_order = [
        "ticker", "expiry", "strike", "bid", "ask", "mid",
        "lastPrice", "volume", "openInterest", "spot"
    ]
    df_all = df_all[col_order]

    # Save
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"\nRaw dataset saved to: {OUTPUT_PATH}")

    print_summary(df_all)


if __name__ == "__main__":
    main()
