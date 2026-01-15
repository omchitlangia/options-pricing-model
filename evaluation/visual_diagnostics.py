import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load data
# -------------------------------
priced_path = "data/processed/spy_priced_bs.csv"
iv_path = "data/processed/spy_with_implied_vol.csv"

df_priced = pd.read_csv(priced_path)
df_iv = pd.read_csv(iv_path)

# -------------------------------
# Plot 1 — BS pricing error vs moneyness
# -------------------------------
plt.figure()
plt.scatter(df_priced["moneyness"], df_priced["error"], alpha=0.6)
plt.axhline(0)
plt.xlabel("Moneyness (S / K)")
plt.ylabel("Pricing Error (BS − Market)")
plt.title("Black–Scholes Pricing Error vs Moneyness")
plt.show()

# -------------------------------
# Plot 2 — BS pricing error vs maturity
# -------------------------------
plt.figure()
plt.scatter(df_priced["T"], df_priced["error"], alpha=0.6)
plt.axhline(0)
plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Pricing Error (BS − Market)")
plt.title("Black–Scholes Pricing Error vs Maturity")
plt.show()

# -------------------------------
# Plot 3 — Implied volatility vs moneyness
# -------------------------------
plt.figure()
plt.scatter(df_iv["moneyness"], df_iv["implied_vol"], alpha=0.6)
plt.xlabel("Moneyness (S / K)")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility vs Moneyness")
plt.show()

# -------------------------------
# Plot 4 — Implied volatility vs maturity
# -------------------------------
plt.figure()
plt.scatter(df_iv["T"], df_iv["implied_vol"], alpha=0.6)
plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility vs Maturity")
plt.show()