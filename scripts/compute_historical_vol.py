import yfinance as yf
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
TICKER = "SPY"
LOOKBACK_DAYS = 90
TRADING_DAYS = 252

# -------------------------------
# Fetch historical prices
# -------------------------------
ticker = yf.Ticker(TICKER)
hist = ticker.history(period=f"{LOOKBACK_DAYS}d")

# -------------------------------
# Compute log returns
# -------------------------------
log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

# -------------------------------
# Annualized volatility
# -------------------------------
sigma = log_returns.std() * np.sqrt(TRADING_DAYS)

print(f"Historical volatility (annualized): {sigma:.4f}")
