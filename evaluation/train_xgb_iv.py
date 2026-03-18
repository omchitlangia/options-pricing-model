import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

DATA_PATH = "data/processed/options_surface/options_surface_filtered.csv"
PLOTS_DIR = "plots/xgb/"

FEATURES = ["log_moneyness", "time_to_maturity", "sqrt_T", "moneyness_T_interaction"]
TARGET   = "implied_vol"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

df_full = df[FEATURES + [TARGET, "moneyness"]].dropna()
assert len(df_full) > 0, "Dataset is empty after dropping missing values."
print(f"Dataset size: {len(df_full)} rows")

X = df_full[FEATURES]
y = df_full[TARGET]

# Same split as all previous models for fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

moneyness        = df_full.loc[X_test.index, "moneyness"].values
time_to_maturity = X_test["time_to_maturity"].values

# ── Train ─────────────────────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred_xgb = model.predict(X_test)
mae_xgb    = mean_absolute_error(y_test, y_pred_xgb)
error_xgb  = y_pred_xgb - y_test.values

# ── Results ───────────────────────────────────────────────────────────────────
print("\nXGBoost Results")
print(f"IV MAE: {mae_xgb:.6f}")

print("\nFirst 5 Predictions vs Actuals:")
sample = pd.DataFrame({"actual": y_test.values[:5], "predicted": y_pred_xgb[:5]})
print(sample.to_string(index=False))

# ── Create plots directory ────────────────────────────────────────────────────
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
ax.scatter(y_test, y_pred_xgb, **STYLE, color="darkorchid", label="Test points")
lims = [min(y_test.min(), y_pred_xgb.min()), max(y_test.max(), y_pred_xgb.max())]
ax.plot(lims, lims, "k--", lw=1, label="Perfect fit")
ax.set_xlabel("Actual IV")
ax.set_ylabel("Predicted IV")
ax.set_title("Actual vs Predicted IV — XGBoost")
ax.legend(fontsize=8)
save(fig, "xgb_actual_vs_pred.png")

# ── Plot 2 — Error vs Moneyness ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(moneyness, error_xgb, **STYLE, color="tomato")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Moneyness")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("XGB Error vs Moneyness")
save(fig, "xgb_error_moneyness.png")

# ── Plot 3 — Error vs Maturity ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(time_to_maturity, error_xgb, **STYLE, color="mediumpurple")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Time to Maturity (years)")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("XGB Error vs Maturity")
save(fig, "xgb_error_maturity.png")

# ── Plot 4 — Volatility Smile Fit ────────────────────────────────────────────
sort_idx = np.argsort(moneyness)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(moneyness[sort_idx], y_test.values[sort_idx],
           s=10, alpha=0.4, color="black", label="Actual IV")
ax.scatter(moneyness[sort_idx], y_pred_xgb[sort_idx],
           s=10, alpha=0.5, color="darkorchid", label="XGB prediction")
ax.set_xlabel("Moneyness")
ax.set_ylabel("Implied Volatility")
ax.set_title("Volatility Smile — XGBoost")
ax.legend(fontsize=8)
save(fig, "xgb_smile.png")

# ── Feature Importance ────────────────────────────────────────────────────────
print("\nFeature Importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat} → {imp:.4f}")

# ── Interpretation ────────────────────────────────────────────────────────────
print("\n── Interpretation ──────────────────────────────────────────────")
u_shape = np.corrcoef(moneyness, np.abs(error_xgb))[0, 1]
print(f"Error correlation with |moneyness|: {u_shape:.3f}")
print(f"Residual std: {np.std(error_xgb):.4f}")

if mae_xgb < 0.0635:
    print(f"  → XGB MAE ({mae_xgb:.4f}) improves over RF (0.0635).")
else:
    print(f"  → XGB MAE ({mae_xgb:.4f}) does not improve over RF (0.0635).")

if abs(u_shape) < 0.05:
    print("  → Error is flat across moneyness — wings well captured.")
elif abs(u_shape) < abs(0.135):
    print("  → Error is flatter than RF across moneyness — better wing capture.")
else:
    print("  → Moneyness-dependent error persists at similar level to RF.")

print("  → Gradient boosting builds trees sequentially, each correcting")
print("    residuals from the prior round — captures finer structure than")
print("    bagging (RF), especially in high-IV regions.")
print("  → subsample=0.8 and colsample_bytree=0.8 provide stochastic")
print("    regularization; reg_lambda=1.0 adds L2 penalty on leaf weights.")
print("────────────────────────────────────────────────────────────────")
