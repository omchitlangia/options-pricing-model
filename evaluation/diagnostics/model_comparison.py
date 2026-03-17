import pandas as pd

bs = pd.read_csv("data/processed/spy_priced_bs.csv")
bi = pd.read_csv("data/processed/spy_priced_binomial.csv")
mc = pd.read_csv("data/processed/spy_priced_mc.csv")

df = bs[["moneyness", "T", "error"]].copy()
df = df.rename(columns={"error": "bs_error"})
df["binomial_error"] = bi["binomial_error"]
df["mc_error"] = mc["mc_error"]

print("Mean Abs Errors")
print("BS:", df["bs_error"].abs().mean())
print("Binomial:", df["binomial_error"].abs().mean())
print("Monte Carlo:", df["mc_error"].abs().mean())

# -------------------------
# Moneyness structure
# -------------------------
bins = pd.cut(df["moneyness"], [0, 0.97, 1.03, 10], labels=["OTM","ATM","ITM"])

print("\nBy Moneyness (mean error)")
print(df.groupby(bins)[["bs_error","binomial_error","mc_error"]].mean())

# -------------------------
# Maturity structure
# -------------------------
# -------------------------
# Safe maturity bins
# -------------------------
Tmax = df["T"].max()

if Tmax < 0.10:
    tbins = pd.cut(
        df["T"],
        [0, 0.05, Tmax + 1e-6],
        labels=["Short","Medium"]
    )
else:
    tbins = pd.cut(
        df["T"],
        [0, 0.05, 0.10, Tmax + 1e-6],
        labels=["Short","Medium","Long"]
    )

print("\nBy Maturity (mean error)")
print(df.groupby(tbins)[["bs_error","binomial_error","mc_error"]].mean())

