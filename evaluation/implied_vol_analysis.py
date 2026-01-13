import pandas as pd

INPUT_PATH = "data/processed/spy_with_implied_vol.csv"

df = pd.read_csv(INPUT_PATH)

# -------------------------------
# Moneyness bins (call options)
# -------------------------------
moneyness_bins = pd.cut(
    df["moneyness"],
    bins=[0.85, 0.95, 1.05, 1.15],
    labels=["OTM", "ATM", "ITM"]
)

vol_by_moneyness = (
    df.groupby(moneyness_bins, observed=True)["implied_vol"]
      .mean()
)

print("Average implied volatility by moneyness:")
print(vol_by_moneyness)

# -------------------------------
# Time-to-maturity bins
# -------------------------------
T_max = df["T"].max()
print(f"\nT_max = {T_max}")

if T_max > 0.10:
    bins = [0.0, 0.05, 0.10, T_max]
    labels = ["Short", "Medium", "Long"]
else:
    bins = [0.0, 0.05, T_max]
    labels = ["Short", "Medium"]

time_bins = pd.cut(
    df["T"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

vol_by_maturity = (
    df.groupby(time_bins, observed=True)["implied_vol"]
      .mean()
)

print("\nAverage implied volatility by maturity:")
print(vol_by_maturity)