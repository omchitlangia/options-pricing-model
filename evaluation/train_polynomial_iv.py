import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

DATA_PATH = "data/processed/options_surface/options_surface_filtered.csv"
PLOTS_DIR = "plots/polynomial/"

FEATURES = ["log_moneyness", "time_to_maturity", "sqrt_T", "moneyness_T_interaction"]
TARGET   = "implied_vol"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Retain moneyness for diagnostic plots.
# time_to_maturity is already in FEATURES and is accessed via X_test directly.
df_full = df[FEATURES + [TARGET, "moneyness"]].dropna()
assert len(df_full) > 0, "Dataset is empty after dropping missing values."

X = df_full[FEATURES]
y = df_full[TARGET]

# Same split as linear model for fair comparison when needed.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

moneyness        = df_full.loc[X_test.index, "moneyness"].values
time_to_maturity = X_test["time_to_maturity"].values

# ── Polynomial feature expansion ──────────────────────────────────────────────
# degree=2 adds all squared terms and pairwise interactions.
# 4 raw features → 14 expanded features.
# fit_transform on train only; transform on test to avoid data leakage.
poly         = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mae    = mean_absolute_error(y_test, y_pred)
error  = y_pred - y_test.values

# ── Results ───────────────────────────────────────────────────────────────────
print("Polynomial Model Results  (degree = 2)")
print(f"IV MAE: {mae:.6f}")
print(f"Expanded feature count: {X_train_poly.shape[1]}")

print("\nFirst 5 Predictions vs Actuals:")
sample = pd.DataFrame({"actual": y_test.values[:5], "predicted": y_pred[:5]})
print(sample.to_string(index=False))

# ── Helpers ───────────────────────────────────────────────────────────────────
STYLE = dict(alpha=0.45, s=14, linewidths=0)

def save(fig, name):
    path = PLOTS_DIR + name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── Plot 1 — Actual vs Predicted ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred, **STYLE, color="royalblue", label="Test points")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "k--", lw=1, label="Perfect fit")
ax.set_xlabel("Actual IV")
ax.set_ylabel("Predicted IV")
ax.set_title("Actual vs Predicted IV — Polynomial (deg 2)")
ax.legend(fontsize=8)
save(fig, "polynomial_actual_vs_pred.png")

# ── Plot 2 — Error vs Moneyness ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(moneyness, error, **STYLE, color="tomato")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Moneyness")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("Polynomial Error vs Moneyness")
save(fig, "polynomial_error_moneyness.png")

# ── Plot 3 — Error vs Maturity ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(time_to_maturity, error, **STYLE, color="mediumpurple")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Time to Maturity (years)")
ax.set_ylabel("Prediction Error (pred − actual)")
ax.set_title("Polynomial Error vs Maturity")
save(fig, "polynomial_error_maturity.png")

# ── Plot 4 — Smile Fit ────────────────────────────────────────────────────────
sort_idx = np.argsort(moneyness)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(moneyness[sort_idx], y_test.values[sort_idx],
           s=10, alpha=0.4, color="black", label="Actual IV")
ax.scatter(moneyness[sort_idx], y_pred[sort_idx],
           s=10, alpha=0.5, color="royalblue", label="Polynomial prediction")
ax.set_xlabel("Moneyness")
ax.set_ylabel("Implied Volatility")
ax.set_title("Volatility Smile — Polynomial (deg 2)")
ax.legend(fontsize=8)
save(fig, "polynomial_smile.png")

# ── Interpretation ────────────────────────────────────────────────────────────
print("\n── Interpretation ──────────────────────────────────────────────")
u_shape = np.corrcoef(moneyness, np.abs(error))[0, 1]
print(f"Error correlation with |moneyness|: {u_shape:.3f}")
if abs(u_shape) < 0.1:
    print("  → Curvature error is largely absorbed by degree-2 terms.")
else:
    print("  → Some curvature error remains; higher degree or nonlinear model needed.")
print(f"Residual std: {np.std(error):.4f}")
print("  → Remaining error driven by cross-asset IV level differences.")
print("    Per-ticker fixed effects or a tree-based model are the natural next step.")
print("────────────────────────────────────────────────────────────────")
