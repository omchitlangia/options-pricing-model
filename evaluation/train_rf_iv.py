import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/processed/options_surface/options_surface_filtered.csv"
PLOTS_DIR = "plots/rf/"

FEATURES = ["log_moneyness", "time_to_maturity", "sqrt_T", "moneyness_T_interaction"]
TARGET   = "implied_vol"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

df_full = df[FEATURES + [TARGET, "moneyness"]].dropna()
assert len(df_full) > 0, "Dataset is empty after dropping missing values."
print(f"Dataset size: {len(df_full)} rows")

X = df_full[FEATURES]
y = df_full[TARGET]

# Same split as linear and polynomial models for fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

moneyness        = df_full.loc[X_test.index, "moneyness"].values
time_to_maturity = X_test["time_to_maturity"].values

# ── Train ─────────────────────────────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred_rf = model.predict(X_test)
mae_rf    = mean_absolute_error(y_test, y_pred_rf)
error_rf  = y_pred_rf - y_test.values

# ── Results ───────────────────────────────────────────────────────────────────
print("\nRandom Forest Results")
print(f"IV MAE: {mae_rf:.6f}")

print("\nFirst 5 Predictions vs Actuals:")
sample = pd.DataFrame({"actual": y_test.values[:5], "predicted": y_pred_rf[:5]})
print(sample.to_string(index=False))

# ── Create plots directory ────────────────────────────────────────────────────
import os
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
STYLE = dict(alpha=0.45, s=14, linewidths=0)

def save(fig, name):
    path = PLOTS_DIR + name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── Plot 1 — Actual vs Predicted ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred_rf, **STYLE, color="forestgreen", label="Test points")
lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
ax.plot(lims, lims, "k--", lw=1, label="Perfect fit")
ax.set_xlabel("Actual IV")
ax.set_ylabel("Predicted IV")
ax.set_title("Actual vs Predicted IV — Random Forest")
ax.legend(fontsize=8)
save(fig, "rf_actual_vs_pred.png")

# ── Plot 2 — Error vs Moneyness ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(moneyness, error_rf, **STYLE, color="tomato")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Moneyness")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("RF Error vs Moneyness")
save(fig, "rf_error_moneyness.png")

# ── Plot 3 — Error vs Maturity ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(time_to_maturity, error_rf, **STYLE, color="mediumpurple")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Time to Maturity (years)")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("RF Error vs Maturity")
save(fig, "rf_error_maturity.png")

# ── Plot 4 — Volatility Smile Fit ────────────────────────────────────────────
sort_idx = np.argsort(moneyness)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(moneyness[sort_idx], y_test.values[sort_idx],
           s=10, alpha=0.4, color="black", label="Actual IV")
ax.scatter(moneyness[sort_idx], y_pred_rf[sort_idx],
           s=10, alpha=0.5, color="forestgreen", label="RF prediction")
ax.set_xlabel("Moneyness")
ax.set_ylabel("Implied Volatility")
ax.set_title("Volatility Smile — Random Forest")
ax.legend(fontsize=8)
save(fig, "rf_smile.png")

# ── Feature Importance ────────────────────────────────────────────────────────
print("\nFeature Importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat} → {imp:.4f}")

# ── Interpretation ────────────────────────────────────────────────────────────
print("\n── Interpretation ──────────────────────────────────────────────")
u_shape = np.corrcoef(moneyness, np.abs(error_rf))[0, 1]
print(f"Error correlation with |moneyness|: {u_shape:.3f}")
print(f"Residual std: {np.std(error_rf):.4f}")

if mae_rf < 0.0901:
    print(f"  → RF MAE ({mae_rf:.4f}) improves over polynomial (0.0901).")
else:
    print(f"  → RF MAE ({mae_rf:.4f}) does not improve over polynomial (0.0901).")

if abs(u_shape) < 0.05:
    print("  → Error is flat across moneyness — curvature well captured.")
else:
    print("  → Some moneyness-dependent error structure remains.")

print("  → RF captures cross-asset IV differences via tree splits,")
print("    absorbing the level bias that linear/polynomial models cannot.")
if model.feature_importances_.max() > 0.5:
    dominant = FEATURES[np.argmax(model.feature_importances_)]
    print(f"  → Dominant feature: {dominant} — drives most splits.")
print("  → max_depth=8 and min_samples_leaf=5 limit overfitting risk.")
print("────────────────────────────────────────────────────────────────")
