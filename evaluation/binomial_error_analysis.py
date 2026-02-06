import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/spy_priced_binomial.csv")

print("Mean abs error:", df["binomial_error"].abs().mean())

# -------------------------
# Moneyness bins
# -------------------------
moneyness_bins = pd.cut(
    df["moneyness"],
    bins=[0, 0.97, 1.03, 10],
    labels=["OTM", "ATM", "ITM"]
)

print("\nBinomial error by moneyness:")
print(df.groupby(moneyness_bins, observed=False)["binomial_error"].mean())

# -------------------------
# Maturity bins
# -------------------------
T_max = df["T"].max()

if T_max < 0.10:
    bins = [0.0, 0.05, T_max + 1e-6]
    labels = ["Short", "Medium"]
else:
    bins = [0.0, 0.05, 0.10, T_max + 1e-6]
    labels = ["Short", "Medium", "Long"]

time_bins = pd.cut(df["T"], bins=bins, labels=labels)

print("\nBinomial error by maturity:")
print(df.groupby(time_bins)["binomial_error"].mean())

#-------------------------
# Visual diagnostics
#-------------------------

df = pd.read_csv("data/processed/spy_priced_binomial.csv")

plt.figure()
plt.scatter(df["moneyness"], df["binomial_error"], alpha=0.6)
plt.axhline(0)
plt.xlabel("Moneyness")
plt.ylabel("Binomial Error")
plt.title("Binomial Error vs Moneyness")
plt.show()

plt.figure()
plt.scatter(df["T"], df["binomial_error"], alpha=0.6)
plt.axhline(0)
plt.xlabel("Maturity")
plt.ylabel("Binomial Error")
plt.title("Binomial Error vs Maturity")
plt.show()