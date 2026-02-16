import pandas as pd
from models.monte_carlo import monte_carlo_call_price

INPUT_PATH = "data/processed/spy_with_implied_vol.csv"
OUTPUT_PATH = "data/processed/spy_priced_mc.csv"

RISK_FREE_RATE = 0.036
NSIMS = 50000

df = pd.read_csv(INPUT_PATH)

mc_prices = []
mc_errors = []

for _, row in df.iterrows():

    price = monte_carlo_call_price(
        S=row["spot"],
        K=row["strike"],
        T=row["T"],
        r=RISK_FREE_RATE,
        sigma=row["implied_vol"],
        n_sims=NSIMS,
        seed=42   # reproducible
    )

    mc_prices.append(price)
    mc_errors.append(price - row["mid"])

df["mc_price"] = mc_prices
df["mc_error"] = mc_errors

df.to_csv(OUTPUT_PATH, index=False)

print("Monte Carlo pricing complete")
print("Options priced:", len(df))
print("Mean abs error:", df["mc_error"].abs().mean())
