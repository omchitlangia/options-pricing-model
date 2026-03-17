import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load data
# -------------------------------
INPUT_PATH = "data/processed/spy_with_greeks.csv"
df = pd.read_csv(INPUT_PATH)

# -------------------------------
# Delta vs Moneyness
# -------------------------------
plt.figure()
plt.scatter(df["moneyness"], df["delta"], alpha=0.6)
plt.xlabel("Moneyness (S / K)")
plt.ylabel("Delta")
plt.title("Delta vs Moneyness")
plt.show()

# -------------------------------
# Gamma vs Moneyness
# -------------------------------
plt.figure()
plt.scatter(df["moneyness"], df["gamma"], alpha=0.6)
plt.xlabel("Moneyness (S / K)")
plt.ylabel("Gamma")
plt.title("Gamma vs Moneyness")
plt.show()

# -------------------------------
# Vega vs Moneyness
# -------------------------------
plt.figure()
plt.scatter(df["moneyness"], df["vega"], alpha=0.6)
plt.xlabel("Moneyness (S / K)")
plt.ylabel("Vega")
plt.title("Vega vs Moneyness")
plt.show()

# -------------------------------
# Vega vs Maturity
# -------------------------------
plt.figure()
plt.scatter(df["T"], df["vega"], alpha=0.6)
plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Vega")
plt.title("Vega vs Maturity")
plt.show()

# -------------------------------
# Theta vs Maturity (optional)
# -------------------------------
plt.figure()
plt.scatter(df["T"], df["theta"], alpha=0.6)
plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Theta")
plt.title("Theta vs Maturity")
plt.show()

