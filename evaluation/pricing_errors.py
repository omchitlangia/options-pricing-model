import pandas as pd
from models.black_scholes import black_scholes_call

# -------------------------------
# Configuration
# -------------------------------
INPUT_PATH = "data/processed/spy_options_clean.csv"
OUTPUT_PATH = "data/processed/spy_priced_bs.csv"

RISK_FREE_RATE = 0.036 # current 3 months T-bill rate
SIGMA = 0.1132   # historical volatility from STEP 3.3

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(INPUT_PATH)

# -------------------------------
# Price using Black–Scholes
# -------------------------------
df["bs_price"] = df.apply(
    lambda x: black_scholes_call(
        S=x["spot"],
        K=x["strike"],
        T=x["T"],
        r=RISK_FREE_RATE,
        sigma=SIGMA
    ),
    axis=1
)

# -------------------------------
# Pricing errors
# -------------------------------
df["error"] = df["bs_price"] - df["mid"]
df["abs_error"] = df["error"].abs()
df["rel_error"] = df["error"] / df["mid"]

# -------------------------------
# Save results
# -------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("Black–Scholes pricing complete")
print(f"Options priced: {len(df)}")
print("Mean absolute error:", df["abs_error"].mean())
print("Mean relative error:", df["rel_error"].mean())
