import pandas as pd
import matplotlib.pyplot as plt

bs = pd.read_csv("data/processed/spy_priced_bs.csv")
bi = pd.read_csv("data/processed/spy_priced_binomial.csv")
mc = pd.read_csv("data/processed/spy_priced_mc.csv")

df = pd.DataFrame({
    "moneyness": bs["moneyness"],
    "T": bs["T"],
    "bs_error": bs["error"],
    "binomial_error": bi["binomial_error"],
    "mc_error": mc["mc_error"]
})

# -------------------------
# Error vs Moneyness
# -------------------------
plt.figure()
plt.scatter(df["moneyness"], df["bs_error"], alpha=0.5, label="BS")
plt.scatter(df["moneyness"], df["binomial_error"], alpha=0.5, label="Binomial")
plt.scatter(df["moneyness"], df["mc_error"], alpha=0.5, label="Monte Carlo")
plt.axhline(0)
plt.xlabel("Moneyness")
plt.ylabel("Pricing Error")
plt.title("Model Error vs Moneyness")
plt.legend()
plt.show()

# -------------------------
# Error vs Maturity
# -------------------------
plt.figure()
plt.scatter(df["T"], df["bs_error"], alpha=0.5, label="BS")
plt.scatter(df["T"], df["binomial_error"], alpha=0.5, label="Binomial")
plt.scatter(df["T"], df["mc_error"], alpha=0.5, label="Monte Carlo")
plt.axhline(0)
plt.xlabel("Maturity (T)")
plt.ylabel("Pricing Error")
plt.title("Model Error vs Maturity")
plt.legend()
plt.show()
