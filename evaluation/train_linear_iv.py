import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/processed/options_surface/options_surface_filtered.csv"

FEATURES = ["log_moneyness", "time_to_maturity", "sqrt_T", "moneyness_T_interaction"]
TARGET = "implied_vol"

# --- Load data ---
df = pd.read_csv(DATA_PATH)

df = df[FEATURES + [TARGET]].dropna()
assert len(df) > 0, "Dataset is empty after dropping missing values."

# --- Split ---
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# --- Train ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Predict ---
y_pred = model.predict(X_test)

# --- Evaluate ---
mae = mean_absolute_error(y_test, y_pred)
print("Linear Model Results")
print(f"IV MAE: {mae:.6f}")

# --- Coefficients ---
print("\nModel Coefficients:")
for feature, coef in zip(FEATURES, model.coef_):
    print(f"  {feature}: {coef:.6f}")
print(f"  intercept: {model.intercept_:.6f}")

# --- First 5 predictions vs actuals ---
print("\nFirst 5 Predictions vs Actuals:")
results = pd.DataFrame({"actual": y_test.values[:5], "predicted": y_pred[:5]})
print(results.to_string(index=False))
