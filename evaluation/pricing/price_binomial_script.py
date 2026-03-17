import pandas as pd
from models.pricing.binomial import binomial_call_price

INPUT_PATH = "data/processed/spy_with_implied_vol.csv"
OUTPUT_PATH = "data/processed/spy_priced_binomial.csv"

RISK_FREE_RATE = 0.036
STEPS = 200   # tree depth — accuracy vs speed tradeoff

df = pd.read_csv(INPUT_PATH)

prices = []
errors = []

for _, row in df.iterrows():

    price = binomial_call_price(
        S=row["spot"],
        K=row["strike"],
        T=row["T"],
        r=RISK_FREE_RATE,
        sigma=row["implied_vol"],
        steps=STEPS
    )

    prices.append(price)
    errors.append(price - row["mid"])

df["binomial_price"] = prices
df["binomial_error"] = errors

df.to_csv(OUTPUT_PATH, index=False)

print("Binomial pricing complete")
print("Options priced:", len(df))
print("Mean abs error:", df["binomial_error"].abs().mean())
