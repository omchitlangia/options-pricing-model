import pandas as pd
from models.iv.implied_vol_solver import implied_vol_bisection

# -------------------------------
# Configuration
# -------------------------------
INPUT_PATH = "data/processed/spy_priced_bs.csv"
OUTPUT_PATH = "data/processed/spy_with_implied_vol.csv"

RISK_FREE_RATE = 0.036  # market-consistent rate

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(INPUT_PATH)

# -------------------------------
# Compute implied volatility
# -------------------------------
df["implied_vol"] = df.apply(
    lambda x: implied_vol_bisection(
        market_price=x["mid"],
        S=x["spot"],
        K=x["strike"],
        T=x["T"],
        r=RISK_FREE_RATE
    ),
    axis=1
)

# -------------------------------
# Drop invalid results
# -------------------------------
df = df.dropna(subset=["implied_vol"])

# -------------------------------
# Save output
# -------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("Implied volatility computation complete")
print(f"Options processed: {len(df)}")
print("Implied vol summary:")
print(df["implied_vol"].describe())
