import pandas as pd

INPUT_PATH = "data/processed/spy_priced_bs.csv"

df = pd.read_csv(INPUT_PATH)

# -------------------------------
# Bin by moneyness
# -------------------------------
moneyness_bins = pd.cut(
    df["moneyness"],
    bins=[0.85, 0.95, 1.05, 1.15],
    labels=["ITM", "ATM", "OTM"]
)

moneyness_errors = df.groupby(moneyness_bins)["error"].mean()

print("Average pricing error by moneyness:")
print(moneyness_errors)

# -------------------------------
# Bin by time to maturity
# -------------------------------
time_bins = pd.cut(
    df["T"],
    bins=[0, 0.05, 0.10, 0.123],
    labels=["Short", "Medium", "Long"]
)

time_errors = df.groupby(time_bins)["error"].mean()

print("\nAverage pricing error by maturity:")
print(time_errors)
