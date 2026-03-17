import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# -------------------------------
# Configuration
# -------------------------------
TICKER = "SPY"
MIN_DTE = 7
MAX_DTE = 45
OUTPUT_PATH = "data/raw/spy_options_raw.csv"

# -------------------------------
# Fetch underlying
# -------------------------------
ticker = yf.Ticker(TICKER)

spot = ticker.history(period="1d")["Close"].iloc[-1]

today = datetime.now(timezone.utc).date()

rows = []

# -------------------------------
# Fetch option chains
# -------------------------------
for exp in ticker.options:
    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
    dte = (exp_date - today).days

    if MIN_DTE <= dte <= MAX_DTE:
        calls = ticker.option_chain(exp).calls.copy()

        calls["expiry"] = exp_date
        calls["dte"] = dte
        calls["spot"] = spot

        rows.append(calls)

# -------------------------------
# Save raw data
# -------------------------------
df = pd.concat(rows, ignore_index=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved raw options data to {OUTPUT_PATH}")
print(f"Rows collected: {len(df)}")
print(f"Spot price used: {spot:.2f}")
