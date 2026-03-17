"""
compute_historical_vol.py
--------------------------
Computes 90-day annualized historical volatility for SPY.

Run from the project root:
    python -m pipeline.compute_historical_vol
"""

import yfinance as yf
import numpy as np

TICKER = "SPY"
LOOKBACK_DAYS = 90
TRADING_DAYS = 252

ticker = yf.Ticker(TICKER)
hist = ticker.history(period=f"{LOOKBACK_DAYS}d")

log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

sigma = log_returns.std() * np.sqrt(TRADING_DAYS)

print(f"Historical volatility (annualized): {sigma:.4f}")
