# Project Context — Options Pricing Model

## 1. Project Overview

This repository implements a quantitative options pricing research framework built to answer a
single core question:

> **How accurately do classical option pricing models price liquid US equity options, and where
> do their assumptions break down?**

The project works through three classical pricing engines — Black–Scholes, Binomial Tree, and
Monte Carlo — applies them to real market data from SPY (S&P 500 ETF), and systematically
evaluates where and how each model fails. The evaluation is built around implied volatility:
by extracting the volatility the market implies and feeding it back into each model, the project
isolates whether pricing inaccuracies come from the model structure or from the volatility input.

The research proceeds step-by-step: collect real option quotes → clean the data → price with
each model → compute errors → extract implied volatility → reprice with implied vol → observe
the volatility smile.

---

## 2. Pricing Models Implemented

### Black–Scholes (`models/black_scholes.py`)

Closed-form pricing formula for European call options. Inputs: spot price `S`, strike `K`,
time to maturity `T` (in years), risk-free rate `r`, volatility `sigma`. Uses SciPy's
`norm.cdf` for the normal CDF. Handles edge cases: `sigma=0` returns intrinsic value,
`T=0` returns immediate payoff.

Key formula:
`C = S * N(d1) - K * exp(-rT) * N(d2)`
`d1 = [ln(S/K) + (r + 0.5σ²)T] / (σ√T)`
`d2 = d1 − σ√T`

### Binomial Tree (`models/binomial.py`)

Cox-Ross-Rubinstein (CRR) binomial tree for European calls. Defaults to 100 steps (200 used
in evaluation for higher accuracy). Builds terminal prices using up/down factors
`u = exp(σ√dt)`, `d = 1/u`, then performs backward induction with the risk-neutral
probability `p = (exp(r*dt) - d) / (u - d)`. Converges to Black–Scholes as steps → ∞.

### Monte Carlo (`models/monte_carlo.py`)

Risk-neutral GBM terminal simulation for European calls. Draws `n_sims=50000` standard
normal shocks and simulates terminal stock price:
`S_T = S * exp((r − 0.5σ²)T + σ√T * Z)`
Payoffs are `max(S_T − K, 0)`, discounted at the risk-free rate. Uses `seed=42` for
reproducibility. Only simulates the terminal price (not full paths) since European option
payoffs depend only on `S_T`.

### Greeks (`models/greeks.py`)

Analytic Black–Scholes Greeks for call options:
- **Delta**: `N(d1)` — sensitivity to spot price
- **Gamma**: `N'(d1) / (S * σ * √T)` — rate of change of delta
- **Vega**: `S * N'(d1) * √T` — sensitivity to volatility (per unit vol change)
- **Theta**: decay of option value with time

---

## 3. Data Pipeline

The current pipeline is sequential and operates on SPY options only:

```
scripts/pull_spy_options.py
    → data/raw/spy_options_raw.csv

scripts/clean_spy_options.py
    → data/processed/spy_options_clean.csv

evaluation/pricing_errors.py          (BS pricing with constant historical vol)
    → data/processed/spy_priced_bs.csv

evaluation/compute_implied_vol.py      (IV extraction via bisection)
    → data/processed/spy_with_implied_vol.csv

evaluation/price_binomial.py           (Binomial pricing with implied vol)
    → data/processed/spy_priced_binomial.csv

evaluation/price_mc.py                 (Monte Carlo pricing with implied vol)
    → data/processed/spy_priced_mc.csv

evaluation/compute_greeks.py           (Analytic Greeks using implied vol)
    → data/processed/spy_with_greeks.csv

evaluation/error_analysis.py           (Error breakdown by moneyness and maturity)
evaluation/binomial_error_analysis.py
evaluation/model_comparison.py
evaluation/implied_vol_analysis.py

evaluation/visual_diagnostics.py       (Scatter plots)
evaluation/model_comparison_plots.py
evaluation/greek_analysis.py
```

---

## 4. Current Dataset

**Source:** `data/raw/spy_options_raw.csv`
**Filtered to:** `data/processed/spy_options_clean.csv` and downstream files
**Size:** ~355 options (after cleaning)

**Raw columns from yfinance:**
`contractSymbol`, `lastTradeDate`, `strike`, `lastPrice`, `bid`, `ask`, `change`,
`percentChange`, `volume`, `openInterest`, `impliedVolatility`, `inTheMoney`,
`contractSize`, `currency`, `expiry`, `dte`, `spot`

**Added during cleaning:**
- `mid = (bid + ask) / 2`
- `moneyness = spot / strike`
- `T = dte / 365.0`

**Scope constraints:**
- Single underlying: SPY only
- DTE window: 7–45 days (near-term only)
- Strike filter: moneyness between 0.85 and 1.15
- Liquidity filter: `volume >= 10` OR `openInterest >= 100`

**Limitations:**
- Only one ticker — no cross-sectional variation
- Short-dated options only — cannot study term structure
- ~355 rows is too small for regression or ML modeling
- No surface structure: insufficient strike × maturity coverage

---

## 5. Feature Engineering

**Currently computed features (in `scripts/clean_spy_options.py` and `evaluation/pricing_errors.py`):**

| Feature | Formula | Purpose |
|---|---|---|
| `mid` | `(bid + ask) / 2` | Market reference price for all models |
| `moneyness` | `spot / strike` | Relative strike position; >1 = ITM call |
| `T` | `dte / 365.0` | Time to maturity in years |
| `bs_price` | Black–Scholes call | Model price under constant historical vol |
| `error` | `bs_price - mid` | Raw pricing error (model minus market) |
| `abs_error` | `|error|` | Absolute error magnitude |
| `rel_error` | `error / mid` | Relative error as fraction of price |
| `implied_vol` | bisection solver | Volatility that makes BS price = market price |
| `binomial_price` | Binomial call | Price using implied vol in CRR tree |
| `binomial_error` | `binomial_price - mid` | Binomial pricing residual |
| `mc_price` | Monte Carlo call | Price using implied vol in GBM simulation |
| `mc_error` | `mc_price - mid` | Monte Carlo pricing residual |
| `delta`, `gamma`, `vega`, `theta` | Analytic BS Greeks | Sensitivity measures using implied vol |

---

## 6. Evaluation Framework

### Black–Scholes error analysis (`evaluation/pricing_errors.py`, `evaluation/error_analysis.py`)

BS is priced first with a constant historical volatility (`sigma = 0.1132`, computed from
90-day SPY log returns). Errors `bs_price - mid` are then segmented:

- **By moneyness bins:** OTM (0.85–0.95), ATM (0.95–1.05), ITM (1.05–1.15)
- **By maturity bins:** Short (0–5%), Medium (5–10%), Long (>10% of a year)

Pattern observed: systematic underpricing for ATM/ITM options and for longer maturities —
driven by the flat volatility assumption.

### Implied volatility calibration

The bisection solver in `volatility/implied_vol.py` solves for `σ*` such that
`BS(S, K, T, r, σ*) = market_price`. Bounds: `[1e-6, 3.0]`. Tolerance `1e-6`,
up to 100 iterations. Rows where IV cannot be solved (e.g. price below intrinsic) are
dropped. After calibration, pricing errors for all three models collapse near zero —
confirming that pricing accuracy is dominated by the volatility specification.

### Cross-model comparison (`evaluation/model_comparison.py`)

After IV calibration, BS, Binomial, and Monte Carlo prices are compared. All three agree
closely, confirming that the numerical pricing method matters far less than the volatility
input.

---

## 7. Visualizations

**Generated in `evaluation/visual_diagnostics.py` and `evaluation/model_comparison_plots.py`:**

| Plot | Key finding |
|---|---|
| BS error vs moneyness | Systematic underpricing for ATM/ITM; near-zero for deep OTM |
| BS error vs maturity | Error grows with time; flat vol assumption breaks down |
| Implied vol vs moneyness | Clear **volatility smile** — IV higher for OTM and ITM strikes |
| Implied vol vs maturity | **Downward term structure** — short-dated options have higher IV |
| Binomial error vs moneyness | Near-zero residuals post-IV calibration; slight spread at ATM |
| Binomial error vs maturity | No systematic bias; confirms numerical convergence |
| All three models vs moneyness | BS, Binomial, MC errors virtually identical after IV calibration |
| All three models vs maturity | Same conclusion — model choice is secondary to vol input |

**Generated in `evaluation/greek_analysis.py`:**

| Plot | Content |
|---|---|
| Delta vs moneyness | S-curve from ~0 (deep OTM) to ~1 (deep ITM) |
| Gamma vs moneyness | Bell-shaped peak at ATM |
| Vega vs moneyness | Peak near ATM; larger for longer maturities |
| Vega vs maturity | Vega increases with √T |
| Theta vs maturity | Theta becomes less negative as maturity increases |

**Generated in `evaluation/mc_path_plot.py`:**

50 simulated GBM paths over 1 year (252 steps) illustrating the diffusion and spread of
risk-neutral price trajectories.

---

## 8. Current Limitations

1. **Single underlying (SPY only):** No cross-sectional variation in spot price, dividend
   yield, or volatility regime. Cannot study how IV surface shape varies across different
   assets.

2. **Narrow DTE window (7–45 days):** Short-dated options only. The full term structure of
   implied volatility — from very short to 12+ months — is not captured.

3. **Small dataset (~355 rows):** Far too small for regression analysis, ML feature
   importance, or surface interpolation. Models trained on 355 rows will overfit.

4. **Narrow moneyness range (0.85–1.15):** Deep OTM and deep ITM options (where the smile
   is most pronounced) are excluded. Cannot model the full smile curvature.

5. **No surface structure:** True volatility surface modeling requires a grid of
   (strike × maturity) combinations. The current dataset has only 3–5 expiries and a
   limited strike range — insufficient to fit or visualize a 2D surface.

6. **No cross-asset comparison:** The volatility smile shape varies significantly between
   assets (e.g., SPY vs TSLA vs AAPL). Single-asset analysis cannot reveal how different
   risk profiles manifest in the IV surface.

---

## 9. Next Research Phase

The immediate next phase is to build a **large multi-asset options dataset** (~2000–8000 rows)
covering multiple tickers (SPY, QQQ, AAPL, NVDA, TSLA) and a wide range of strikes and
maturities (up to 12 expiries per ticker).

This expanded dataset enables:

- **Volatility surface visualization:** Plot IV as a 2D surface across strike and maturity
  dimensions for each underlying.

- **Smile modeling:** Fit parametric or interpolated models (SVI, SABR, or polynomial) to
  the observed smile shape.

- **Regression and machine learning:** Train models that predict IV from observable features
  (moneyness, time to maturity, ticker). With 2000+ rows, cross-validation and
  generalization become meaningful.

- **Option pricing reconstruction:** Verify that a model trained on IV predictions can
  reproduce market prices with lower error than constant-vol Black–Scholes.

- **Cross-asset surface comparison:** Compare how the IV surface shape differs between a
  low-vol ETF (SPY) and high-vol single stocks (TSLA, NVDA).

The pipeline for this phase is implemented in `scripts/collect_options_surface.py`,
`processing/clean_options_dataset.py`, `processing/compute_surface_features.py`, and
`processing/compute_implied_vols.py`. The resulting dataset is stored under
`data/processed/options_surface/options_surface_dataset.csv`.

---

## 10. Volatility Surface Exploration (Step 8.2)

**Script:** `evaluation/vol_surface_exploration.py`
**Plots:** `plots/vol_surface/`

### Dataset

The multi-asset surface dataset contains **2061 options** across five tickers:
SPY (519), TSLA (616), QQQ (391), NVDA (299), AAPL (236).
Each ticker has 7–8 expiry dates covering maturities from 1 day to ~60 days (0.003–0.164 years).
Visualizations filter to the liquid trading range: moneyness ∈ [0.70, 1.40] and IV < 2.50,
retaining **1672 rows**. The remaining 389 rows contain deep-ITM options from TSLA and AAPL
(moneyness up to 78) and 85 rows where the bisection solver hit its cap of 3.0.

---

### Observed Smile Structure

The IV vs moneyness scatter (`iv_vs_moneyness.png`) reveals a clear **volatility smile**
present across all five assets:

- **IV is lowest near ATM (moneyness ≈ 1.0)** and rises on both the OTM and ITM wings,
  forming a U-shaped or skewed-smile pattern.
- **Single-stock IV levels are substantially higher than index IV:** TSLA and NVDA carry
  mean IV of ~0.59–0.70 in the 0.90–1.10 moneyness range, while SPY and QQQ sit at
  0.24–0.36 — roughly half the level. This reflects the diversification premium in index
  options and the idiosyncratic risk embedded in single-stock options.
- **The smile steepens sharply on the ITM wing (1.10–1.20):** mean IV across all tickers
  reaches 0.63 in this bucket versus 0.30 in the 0.90–1.00 bucket. SPY ITM options
  (moneyness 1.10–1.20) exhibit a mean IV of 0.83 — more than double their ATM level.
- The smile shape is **asymmetric**: the OTM wing (0.80–0.90) carries mean IV of 0.50,
  while the comparable ITM wing (1.10–1.20) carries 0.63, suggesting a rightward tilt
  consistent with demand for upside participation via calls.

**Moneyness bucket summary (all tickers, plot dataset):**

| Moneyness bucket | Mean IV | Interpretation |
|---|---|---|
| 0.80–0.90 | 0.500 | OTM wing — elevated, fear of spot decline |
| 0.90–1.00 | 0.305 | Near ATM — lowest IV in the dataset |
| 1.00–1.10 | 0.394 | ITM — IV begins rising |
| 1.10–1.20 | 0.626 | Deep ITM — sharp IV increase |

---

### Term Structure of Volatility

The IV vs maturity scatter (`iv_vs_maturity.png`) shows **maturity clustering** rather
than a smooth continuous term structure, driven by the discrete set of expiry dates:

- **Short-dated options (T < 0.05 years, under 18 days) carry the highest mean IV at 0.42.**
  This is consistent with near-term event risk and the well-documented volatility term
  structure decay.
- The relationship is **not monotonically declining**: medium-maturity buckets
  (0.05–0.15 and 0.15–0.30 years) show mean IV of 0.47 and 0.46 respectively —
  nearly equal, suggesting the term structure has flattened at these horizons.
- **SPY and QQQ have no options beyond T = 0.047 years** in this dataset, limiting
  term structure analysis for indices. Single stocks (AAPL, NVDA, TSLA) extend to 0.164
  years and show a clear downward slope: AAPL declines from 0.55 (short) to 0.34 (long).
- The maturity density is heavily concentrated in the short end: 1284 of 1672 filtered
  rows fall in the 0–0.05 year bucket. This imbalance must be accounted for in any
  surface model.

**Maturity bucket summary (all tickers, plot dataset):**

| Maturity bucket | Mean IV | n |
|---|---|---|
| 0–0.05 years | 0.419 | 1284 |
| 0.05–0.15 years | 0.470 | 280 |
| 0.15–0.30 years | 0.462 | 108 |

---

### 3D Surface Geometry

The 3D scatter (`iv_surface_3d.png`) confirms that the volatility surface is not flat
in either dimension simultaneously:

- The surface **rises in both the moneyness and maturity directions** away from the
  ATM short-dated corner.
- Distinct **asset-level separation** is visible along the IV axis: TSLA and NVDA occupy
  the high-IV region (0.4–1.0+) while SPY and QQQ cluster at lower levels (0.2–0.5).
- The surface geometry is **nonlinear and irregular** — no simple planar or quadratic
  surface would fit the combined cross-asset data.

---

### Implications for Modeling the Surface

These observations have direct consequences for the next modeling phase:

1. **A single constant volatility is insufficient** — confirmed across both the moneyness
   and maturity dimensions. This was already established for SPY in the single-asset phase;
   the multi-asset dataset makes it universal.

2. **A per-ticker intercept (fixed effect) is necessary.** TSLA and NVDA IV levels are
   roughly 2–3× higher than SPY and QQQ for similar moneyness and maturity. Any model
   that pools all tickers without an asset-level parameter will produce large systematic
   errors.

3. **Nonlinear moneyness terms are required.** The smile is clearly curved, not linear.
   Moneyness alone cannot capture the U-shape; `moneyness²` or `|log_moneyness|` terms
   are needed.

4. **Interaction between moneyness and maturity must be modeled.** The smile steepness
   and curvature change with time to expiry — the surface is not separable into
   independent smile and term structure components. The `moneyness_T_interaction` feature
   already in the dataset addresses this directly.

5. **The short-maturity concentration (76% of data in T < 0.05)** means the model will
   be trained primarily on near-term behavior. Predictions for longer maturities will
   rely on extrapolation and should be treated cautiously.

6. **Machine learning models (random forest, gradient boosting, neural network) are
   natural candidates** given the nonlinearity and cross-asset variation. A linear
   regression baseline should be built first to measure the benefit of nonlinear methods.

---

### Next Step: Step 8.3 — Volatility Surface Modeling

The surface exploration confirms that the data has sufficient richness and structure to
train predictive models. The recommended modeling sequence is:

1. **Baseline:** Linear regression of IV on `moneyness`, `sqrt_T`, ticker dummies
2. **Polynomial extension:** Add `moneyness²`, `moneyness_T_interaction`
3. **Nonlinear model:** Gradient boosted trees or random forest on all surface features
4. **Evaluation:** Compare models by RMSE and R² on a held-out test set
5. **Reconstruction test:** Use predicted IV in Black–Scholes and compare to market prices

---

## 11. IV Surface Filtering Pipeline (Step 8.3 — Data Curation)

**Script:** `pipeline/filter_iv_dataset.py`
**Input:** `data/processed/options_surface/options_surface_dataset.csv` (2061 rows)
**Output:** `data/processed/options_surface/options_surface_filtered.csv` (1546 rows)

### Design principle

Filters are classified as either *data errors / microstructure noise* (hard remove) or
*legitimate market structure* (keep). The volatility smile, skew, and wing elevations
are real features — over-filtering would artificially flatten the surface.

### Filter sequence and results

| Step | Filter | Removed | Rationale |
|---|---|---|---|
| Sanity | bid ≤ 0, ask ≤ 0, bid > ask, mid ≤ 0, mid < $0.01 | 0 | Dataset already clean |
| Liquidity | relative_spread > 1.0; OR zero volume AND zero OI AND spread > 0.3 | 0 | No spread violations; only 1 dead quote with low spread |
| Maturity | time_to_maturity < 0.003 (~1 day) | 66 | Near-expiry gamma explosion makes IV numerically unstable |
| Moneyness | moneyness outside (0.7, 1.3) | 449 | Outside surface modeling range; deep-ITM rows dominated by solver-capped IV = 3.0 |
| IV | implied_vol < 0.01 or ≥ 3.0 (solver cap) | 0 | Already caught by moneyness filter; acts as safety net |

**Total removed: 515 rows (25.0%)**

### Soft signals (flags, not removals)

- `wide_spread_flag`: relative_spread > 0.3 — 67 rows; elevated quote uncertainty
- `short_maturity_flag`: T < 0.02 (~7 days) — 496 rows; valid data, handle carefully
- `wing_flag`: moneyness < 0.85 or > 1.15 — 313 rows; smile wings, must be retained

### Weighting scheme

`weight = (1 / (1 + relative_spread)) × exp(−|log_moneyness|)`, normalized to [0, 1].

- Base term rewards tight spreads (high market maker confidence)
- ATM boost anchors surface level without discarding wing rows
- Mean weight 0.875; minimum 0.489 (widest-spread wing options)

### IV statistics after filtering

| Stat | Value |
|---|---|
| Mean IV | 0.413 |
| Median IV | 0.355 |
| Std IV | 0.210 |
| Min IV | 0.145 |
| Max IV (winsorized) | 1.301 |

IV is winsorized at the 99th percentile (1.30) for modeling stability.
Original values preserved in `implied_vol_raw`.

### Rows per ticker after filtering

SPY 469 · TSLA 376 · QQQ 350 · NVDA 177 · AAPL 174

---

## 12. Step 8.3 — Model Development (Structured)

### Design rationale: separation of concerns

Each model lives in its own script with no cross-dependencies. This enforces:
- Independent reproducibility: either script runs standalone without the other
- Fair comparison: both scripts use identical data loading, split parameters
  (`test_size=0.20, random_state=42`), and feature set, so MAE differences reflect
  only model capacity, not implementation differences
- Clean extension path: adding a new model (e.g. gradient boosted trees) means
  adding one file, not modifying an existing one

---

### Linear baseline — `evaluation/train_linear_iv.py`

**Input:** `data/processed/options_surface/options_surface_filtered.csv`
**Split:** 80/20, random_state = 42

| Feature | Coefficient |
|---|---|
| `log_moneyness` | +0.486 |
| `time_to_maturity` | +5.736 |
| `sqrt_T` | −2.015 |
| `moneyness_T_interaction` | −1.259 |
| intercept | +0.606 |

**IV MAE: 0.1602**

The dominant error is not smile curvature (error correlation with |moneyness|: 0.083)
but cross-asset IV level bias: TSLA/NVDA carry 2–3× higher IV than SPY/QQQ at the
same moneyness and maturity, which a pooled linear model cannot distinguish.
Residual std: 0.201.

**Plots → `plots/linear/`**

| File | Content |
|---|---|
| `linear_actual_vs_pred.png` | Actual vs predicted IV; diagonal shows systematic underprediction at high IV |
| `linear_error_moneyness.png` | Residuals vs moneyness; level bias is spread-uniform, not U-shaped |
| `linear_error_maturity.png` | Residuals vs time to maturity |
| `linear_smile.png` | Actual IV vs linear fit sorted by moneyness |

---

### Polynomial model — `evaluation/train_polynomial_iv.py`

**Degree:** 2 via `sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)`
**Feature expansion:** 4 raw features → 14 degree-2 terms (squares + pairwise interactions)
**Fit on train only**; transform applied to test to prevent data leakage.

**IV MAE: 0.0901** (−43.8% vs linear)

The squared and cross terms absorb nonlinear smile curvature and the term structure's
√T scaling. Error correlation with |moneyness|: 0.116 — some curvature residual remains,
suggesting the smile is not fully captured by degree-2 terms alone. Residual std: 0.119.
Remaining error is still dominated by the cross-asset IV level gap.

**Plots → `plots/polynomial/`**

| File | Content |
|---|---|
| `polynomial_actual_vs_pred.png` | Tighter scatter around diagonal vs linear |
| `polynomial_error_moneyness.png` | Reduced but non-zero curvature residual |
| `polynomial_error_maturity.png` | Residuals vs time to maturity |
| `polynomial_smile.png` | Actual IV vs polynomial fit sorted by moneyness |

---

### Step 8.3.3 — Random Forest Model (`evaluation/train_rf_iv.py`)

**Motivation:** Linear and polynomial models are constrained to parametric functional
forms. A random forest is a non-parametric ensemble that can capture arbitrary
nonlinear interactions — including cross-asset IV level differences — without explicit
feature engineering (e.g. ticker dummies). This tests whether the IV surface has
structure that parametric models miss.

**Parameters:** `n_estimators=200`, `max_depth=8`, `min_samples_leaf=5`,
`random_state=42`, `n_jobs=-1`

**IV MAE: 0.0635** (−29.5% vs polynomial, −60.4% vs linear)

The RF absorbs cross-asset IV level bias through tree splits on `log_moneyness` and
the interaction term, without needing per-ticker fixed effects. Error correlation with
|moneyness|: 0.135 — some wing structure remains, but residual std drops from 0.119
(polynomial) to 0.090.

**Feature importances:**

| Feature | Importance |
|---|---|
| `log_moneyness` | 0.597 |
| `moneyness_T_interaction` | 0.279 |
| `time_to_maturity` | 0.064 |
| `sqrt_T` | 0.059 |

`log_moneyness` dominates (60% of splits), confirming that the smile's moneyness
dimension carries the most predictive signal. The interaction term captures how smile
shape varies with maturity. The two time features contribute modestly — consistent
with the dataset's heavy concentration in short maturities.

**Regularization:** `max_depth=8` and `min_samples_leaf=5` prevent the forest from
memorizing individual data points. With 1546 rows and 200 trees, each leaf averages
~15 training samples — sufficient for stable estimates without overfitting.

**Plots → `plots/rf/`**

| File | Content |
|---|---|
| `rf_actual_vs_pred.png` | Tighter clustering around diagonal than polynomial |
| `rf_error_moneyness.png` | Flatter error distribution; reduced wing bias |
| `rf_error_maturity.png` | No systematic maturity-dependent pattern |
| `rf_smile.png` | Actual IV vs RF prediction sorted by moneyness |

---

### Step 8.3.4 — XGBoost Model (`evaluation/train_xgb_iv.py`)

**Motivation:** Random Forest uses bagging — each tree is trained independently on a
bootstrap sample. XGBoost uses gradient boosting — trees are built sequentially, with
each new tree fitting the residuals of the ensemble so far. This sequential error
correction can capture finer structure in the IV surface, particularly in high-IV
regions where RF averaging smooths over sharp transitions.

**Parameters:** `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`,
`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `random_state=42`

**IV MAE: 0.0556** (−12.4% vs RF, −38.3% vs polynomial, −65.3% vs linear)

XGBoost achieves the lowest MAE across all models. The sequential residual correction
refines predictions in regions where RF's parallel averaging leaves residual bias.
Error correlation with |moneyness| drops to 0.112 (from 0.135 in RF), indicating
better wing capture. Residual std: 0.082.

**Feature importances:**

| Feature | Importance |
|---|---|
| `log_moneyness` | 0.317 |
| `time_to_maturity` | 0.308 |
| `sqrt_T` | 0.239 |
| `moneyness_T_interaction` | 0.136 |

Unlike RF where `log_moneyness` dominated at 60%, XGBoost distributes importance more
evenly across all four features. This reflects the sequential learning: early trees
capture the dominant moneyness signal, and later trees exploit the time-related features
to correct remaining residuals. The more balanced distribution suggests XGBoost extracts
more information from each feature dimension.

**Regularization:** `max_depth=4` (shallower than RF's 8) keeps individual trees as
weak learners, relying on the ensemble for expressiveness. `subsample=0.8` and
`colsample_bytree=0.8` introduce stochastic regularization. `reg_lambda=1.0` adds L2
penalty on leaf weights. No signs of overfitting: residual std (0.082) is well below
the training signal magnitude.

**Plots → `plots/xgb/`**

| File | Content |
|---|---|
| `xgb_actual_vs_pred.png` | Tightest clustering around diagonal across all models |
| `xgb_error_moneyness.png` | Flatter error than RF; reduced wing bias |
| `xgb_error_maturity.png` | No systematic maturity-dependent pattern |
| `xgb_smile.png` | Actual IV vs XGB prediction sorted by moneyness |

---

### Step 8.3.5 — Neural Network Model (`evaluation/train_nn_iv.py`)

**Motivation:** Tree-based models (RF, XGBoost) partition the feature space into
axis-aligned rectangles. A neural network learns a continuous, differentiable mapping
from features to IV — potentially producing smoother surface interpolation. This tests
whether a fundamentally different model family can compete on a small dataset (n=1546).

**Architecture:** MLP with two hidden layers (64, 32), ReLU activation, Adam optimizer,
`learning_rate_init=0.001`, `max_iter=500`. Features scaled with `StandardScaler`
(fit on training data only). ~2,273 trainable parameters.

**IV MAE: 0.0743** — worse than RF (0.0635) and XGBoost (0.0556)

The MLP converged in only 27 iterations, suggesting early convergence to a suboptimal
solution. Error correlation with |moneyness| is 0.237 — substantially higher than all
other models — indicating the NN struggles most with deep OTM/ITM wings. Residual std:
0.103.

**Why the NN underperforms:** With only 1546 samples and 4 features, tree-based models
have a structural advantage: they can partition the moneyness-maturity space into
fine-grained bins and memorize local IV levels. The MLP must learn the entire surface
through a smooth parametric function, and the limited data provides insufficient signal
for the network to discover the complex nonlinear relationships that trees capture via
splits. Additionally, the volatility surface has sharp transitions (e.g., near-ATM
curvature changes) that ReLU networks need more depth or width to approximate well.

**Plots → `plots/nn/`**

| File | Content |
|---|---|
| `nn_actual_vs_pred.png` | Wider scatter around diagonal than tree models |
| `nn_error_moneyness.png` | Visible moneyness-dependent error pattern |
| `nn_error_maturity.png` | Mild maturity-dependent structure |
| `nn_smile.png` | Actual IV vs NN prediction sorted by moneyness |

---

### Model comparison summary

| Model | IV MAE | Residual std | Error–moneyness corr |
|---|---|---|---|
| Linear | 0.1602 | 0.201 | 0.083 |
| Polynomial (deg 2) | 0.0901 | 0.119 | 0.116 |
| Neural Network (MLP) | 0.0743 | 0.103 | 0.237 |
| Random Forest | 0.0635 | 0.090 | 0.135 |
| XGBoost | 0.0556 | 0.082 | 0.112 |

The tree-based models (RF, XGBoost) remain the best performers. The MLP beats polynomial
regression but falls short of RF, demonstrating that on small tabular datasets,
gradient-boosted trees outperform shallow neural networks. XGBoost remains the best model.

---

### Step 8.4 — Model Comparison (`evaluation/model_comparison.py`)

**Purpose:** Unified comparison of all five IV surface models trained in Steps 8.3.1–8.3.5.
All models are recreated with identical parameters and trained on the same split
(`test_size=0.2, random_state=42`) within a single script to guarantee consistency.

**Final ranking (by MAE):**

| Rank | Model | MAE | Residual Std |
|---|---|---|---|
| 1 | XGBoost | 0.0556 | 0.082 |
| 2 | Random Forest | 0.0635 | 0.090 |
| 3 | Neural Network (MLP) | 0.0743 | 0.103 |
| 4 | Polynomial (deg 2) | 0.0901 | 0.119 |
| 5 | Linear | 0.1602 | 0.201 |

**Model progression insights:**

1. **Linear → Polynomial (−43.8%):** The largest single improvement. Degree-2 terms
   capture the smile's quadratic curvature — the fundamental nonlinearity that a
   linear model cannot represent.

2. **Polynomial → RF (−29.5%):** Tree-based partitioning handles local IV variations
   that a global polynomial cannot, especially cross-asset level differences.

3. **RF → XGBoost (−12.4%):** Sequential boosting corrects residual errors left by
   bagging. Diminishing returns indicate the remaining error is increasingly driven
   by missing features (e.g., per-ticker effects) rather than model capacity.

4. **Neural Network:** Beats polynomial but loses to both tree models. On small tabular
   data (n=1546), tree ensembles outperform shallow MLPs — the NN lacks sufficient
   training signal to discover the same fine-grained structure that trees capture
   via axis-aligned splits.

**Limitations:**
- All models use the same 4 features with no ticker-level information — cross-asset
  IV level differences are a persistent error source.
- The dataset (1546 rows) favors tree methods over neural networks.
- No time-series structure is exploited; all models treat observations as i.i.d.

**Plots → `plots/comparison/`**

| File | Content |
|---|---|
| `smile_comparison.png` | All model predictions overlaid on actual IV vs moneyness |
| `error_moneyness_comparison.png` | Error scatter for all models vs moneyness |
| `actual_vs_pred_comparison.png` | Actual vs predicted IV for all models |

---

### Plot directory structure

```
plots/
    linear/          ← linear model diagnostics only
    polynomial/      ← polynomial model diagnostics only
    rf/              ← random forest diagnostics only
    xgb/             ← XGBoost diagnostics only
    nn/              ← neural network diagnostics only
    comparison/      ← cross-model comparison plots
```

### Next step

Options for further work: (1) reconstruction test — feed predicted IV into Black–Scholes
and compare repriced options to market prices, (2) add per-ticker features to capture
cross-asset IV differences, (3) deeper NN architectures or regularization tuning,
or (4) time-series aware modeling.
