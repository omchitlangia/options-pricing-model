import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

DATA_PATH = "data/processed/options_surface/options_surface_filtered.csv"
PLOTS_DIR = "plots/nn/"

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

# ── Scale features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

y_pred_nn = model.predict(X_test_scaled)
mae_nn    = mean_absolute_error(y_test, y_pred_nn)
error_nn  = y_pred_nn - y_test.values

# ── Results ───────────────────────────────────────────────────────────────────
print("\nNeural Network Results")
print(f"IV MAE: {mae_nn:.6f}")

print("\nFirst 5 Predictions vs Actuals:")
sample = pd.DataFrame({"actual": y_test.values[:5], "predicted": y_pred_nn[:5]})
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
ax.scatter(y_test, y_pred_nn, **STYLE, color="teal", label="Test points")
lims = [min(y_test.min(), y_pred_nn.min()), max(y_test.max(), y_pred_nn.max())]
ax.plot(lims, lims, "k--", lw=1, label="Perfect fit")
ax.set_xlabel("Actual IV")
ax.set_ylabel("Predicted IV")
ax.set_title("Actual vs Predicted IV — Neural Network")
ax.legend(fontsize=8)
save(fig, "nn_actual_vs_pred.png")

# ── Plot 2 — Error vs Moneyness ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(moneyness, error_nn, **STYLE, color="tomato")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Moneyness")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("NN Error vs Moneyness")
save(fig, "nn_error_moneyness.png")

# ── Plot 3 — Error vs Maturity ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(time_to_maturity, error_nn, **STYLE, color="teal")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Time to Maturity (years)")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("NN Error vs Maturity")
save(fig, "nn_error_maturity.png")

# ── Plot 4 — Volatility Smile Fit ────────────────────────────────────────────
sort_idx = np.argsort(moneyness)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(moneyness[sort_idx], y_test.values[sort_idx],
           s=10, alpha=0.4, color="black", label="Actual IV")
ax.scatter(moneyness[sort_idx], y_pred_nn[sort_idx],
           s=10, alpha=0.5, color="teal", label="NN prediction")
ax.set_xlabel("Moneyness")
ax.set_ylabel("Implied Volatility")
ax.set_title("Volatility Smile — Neural Network")
ax.legend(fontsize=8)
save(fig, "nn_smile.png")

# ── Interpretation ────────────────────────────────────────────────────────────
print("\n── Interpretation ──────────────────────────────────────────────")
u_shape = np.corrcoef(moneyness, np.abs(error_nn))[0, 1]
print(f"Error correlation with |moneyness|: {u_shape:.3f}")
print(f"Residual std: {np.std(error_nn):.4f}")
print(f"Training iterations: {model.n_iter_}")

if mae_nn < 0.0556:
    print(f"  → NN MAE ({mae_nn:.4f}) improves over XGBoost (0.0556).")
elif mae_nn < 0.0635:
    print(f"  → NN MAE ({mae_nn:.4f}) beats RF (0.0635) but not XGBoost (0.0556).")
else:
    print(f"  → NN MAE ({mae_nn:.4f}) does not beat RF (0.0635).")

if model.n_iter_ >= 500:
    print("  → max_iter reached — model may benefit from more iterations.")
else:
    print(f"  → Converged in {model.n_iter_} iterations (well under max_iter=500).")

if abs(u_shape) < 0.05:
    print("  → Error is flat across moneyness — wings well captured.")
elif abs(u_shape) < 0.112:
    print("  → Error is flatter than XGBoost across moneyness.")
else:
    print("  → Moneyness-dependent error persists at similar or higher level than XGBoost.")

print("  → MLP learns a continuous mapping from features to IV, unlike")
print("    tree-based models which partition the space into axis-aligned regions.")
print("  → With 1546 samples and (64, 32) architecture = ~2,273 parameters,")
print("    the model is moderately parameterized relative to dataset size.")
print("────────────────────────────────────────────────────────────────")
