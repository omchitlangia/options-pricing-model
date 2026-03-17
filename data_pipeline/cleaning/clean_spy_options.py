import pandas as pd

INPUT_PATH = "data/raw/spy_options_raw.csv"
OUTPUT_PATH = "data/processed/spy_options_clean.csv"

# -------------------------------
# Load raw data
# -------------------------------
df = pd.read_csv(INPUT_PATH)

# -------------------------------
# Compute mid price
# -------------------------------
df["mid"] = (df["bid"] + df["ask"]) / 2

# -------------------------------
# Compute moneyness and time
# -------------------------------
df["moneyness"] = df["spot"] / df["strike"]
df["T"] = df["dte"] / 365.0

# -------------------------------
# Apply filters
# -------------------------------
df_clean = df[
    (df["bid"] > 0) &
    (df["ask"] > df["bid"]) &
    (
        (df["volume"] >= 10) |
        (df["openInterest"] >= 100)
    ) &
    (df["moneyness"].between(0.85, 1.15))
].copy()

# -------------------------------
# Save processed data
# -------------------------------
df_clean.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned data saved to {OUTPUT_PATH}")
print(f"Rows before cleaning: {len(df)}")
print(f"Rows after cleaning: {len(df_clean)}")
