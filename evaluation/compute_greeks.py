import pandas as pd
from models.greeks import delta_call, gamma_call, vega_call, theta_call

INPUT_PATH = "data/processed/spy_with_implied_vol.csv"
OUTPUT_PATH = "data/processed/spy_with_greeks.csv"

RISK_FREE_RATE = 0.036

df = pd.read_csv(INPUT_PATH)

df["delta"] = df.apply(
    lambda x: delta_call(x["spot"], x["strike"], x["T"], RISK_FREE_RATE, x["implied_vol"]),
    axis=1
)

df["gamma"] = df.apply(
    lambda x: gamma_call(x["spot"], x["strike"], x["T"], RISK_FREE_RATE, x["implied_vol"]),
    axis=1
)

df["vega"] = df.apply(
    lambda x: vega_call(x["spot"], x["strike"], x["T"], RISK_FREE_RATE, x["implied_vol"]),
    axis=1
)

df["theta"] = df.apply(
    lambda x: theta_call(x["spot"], x["strike"], x["T"], RISK_FREE_RATE, x["implied_vol"]),
    axis=1
)

df.to_csv(OUTPUT_PATH, index=False)

print("Greeks computed successfully")
print(df[["delta", "gamma", "vega", "theta"]].describe())
