import pandas as pd

INPUT_PATH = "data/processed/spy_priced_bs.csv"

df = pd.read_csv(INPUT_PATH)

# =====================================================
# 1. Moneyness bins (CORRECT for CALL options)
# =====================================================
# Call option:
# OTM : S < K  -> S/K < 1
# ATM : S ≈ K
# ITM : S > K  -> S/K > 1

moneyness_bins = pd.cut(
    df["moneyness"],
    bins=[0.85, 0.95, 1.05, 1.15],
    labels=["OTM", "ATM", "ITM"]
)

moneyness_errors = (
    df.groupby(moneyness_bins, observed=True)["error"]
      .mean()
)

print("Average pricing error by moneyness:")
print(moneyness_errors)

# =====================================================
# 2. Time-to-maturity bins (ROBUST)
# =====================================================
# Do NOT hard-code upper bound
# Compute T_max safely

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

time_errors = (
    df.groupby(time_bins, observed=True)["error"]
      .mean()
)

print("\nAverage pricing error by maturity:")
print(time_errors)