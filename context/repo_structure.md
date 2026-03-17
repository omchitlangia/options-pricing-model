# Repository Structure

Options pricing research codebase — classical models, IV extraction, and volatility surface modeling.
Designed for clean extensibility into ML-based surface models (Step 8.3+).

---

## Architecture Overview

The codebase follows a strict **layered architecture**:

```
config/          ← single source of truth for all parameters and paths
utils/           ← shared I/O and logging utilities
core/            ← pure mathematical engines (no I/O, no side effects)
data_pipeline/   ← all data acquisition, cleaning, and feature engineering
evaluation/      ← reusable analysis functions (pricing, IV, diagnostics)
models/          ← ML surface models (Step 8.3+)
experiments/     ← isolated, reproducible entry points for each step
data/            ← raw and processed datasets
plots/           ← saved figures
context/         ← project documentation
notes/           ← step-by-step implementation notes
```

Each layer depends only on the layers below it:

```
experiments
    └── evaluation  ──┐
    └── data_pipeline ├── core
    └── models        ┘    └── config / utils
```

---

## Directory Tree

```
Options Pricing Model/
│
├── config/
│   └── constants.py            Single source of truth for all parameters and paths
│
├── utils/
│   ├── io.py                   load_dataset / save_dataset helpers
│   └── logging.py              print_header / print_section formatting
│
├── core/                       Pure mathematical engines — no I/O, no file paths
│   ├── pricing/
│   │   ├── black_scholes.py    Closed-form BS call: black_scholes_call(S, K, T, r, sigma)
│   │   ├── binomial.py         CRR binomial tree: binomial_call_price(...)
│   │   └── monte_carlo.py      GBM simulation: monte_carlo_call_price(...)
│   ├── iv/
│   │   └── implied_vol_solver.py  Bisection: implied_vol_bisection(price, S, K, T, r)
│   ├── greeks/
│   │   └── greeks.py           Analytical Greeks: delta, gamma, vega, theta
│   └── math/
│       └── distributions.py    normal_cdf / normal_pdf (centralised scipy wrapper)
│
├── data_pipeline/              Data acquisition, cleaning, and feature engineering
│   ├── ingestion/
│   │   ├── pull_spy_options.py          Fetch SPY option chains from yfinance
│   │   ├── collect_options_surface.py   Fetch 5-ticker surface data
│   │   └── compute_historical_vol.py    Compute annualised historical vol
│   ├── cleaning/
│   │   ├── clean_spy_options.py         SPY: mid price, moneyness, liquidity filters
│   │   └── clean_options_dataset.py     Multi-asset: spread filters, time_to_maturity
│   ├── features/
│   │   └── compute_surface_features.py  moneyness, log_moneyness, sqrt_T, interaction
│   └── iv/
│       └── compute_implied_vols.py      Bisection IV for entire surface dataset
│
├── evaluation/                 Reusable analysis functions (accept DataFrames, return DataFrames)
│   ├── pricing/
│   │   ├── evaluate_pricing_models.py   price_with_bs / price_with_binomial / price_with_mc
│   │   ├── price_bs_script.py           Legacy script (superseded by experiments/)
│   │   ├── price_binomial_script.py     Legacy script (superseded by experiments/)
│   │   └── price_mc_script.py           Legacy script (superseded by experiments/)
│   ├── iv/
│   │   ├── evaluate_iv_models.py        compute_implied_vols / compute_greeks
│   │   ├── compute_implied_vol.py       Legacy script
│   │   └── compute_greeks.py            Legacy script
│   └── diagnostics/
│       ├── error_analysis.py            error_by_moneyness / error_by_maturity / print_error_report
│       ├── vol_surface_exploration.py   Full multi-asset surface analysis + 3D plots
│       ├── model_comparison.py          Side-by-side MAE across BS / Binomial / MC
│       ├── model_comparison_plots.py    Error scatter plots for all three models
│       ├── bs_error_analysis.py         BS error segments (legacy)
│       ├── binomial_error_analysis.py   Binomial error segments (legacy)
│       ├── iv_smile_analysis.py         IV smile and term structure (legacy)
│       ├── greek_analysis.py            Greeks scatter plots (legacy)
│       ├── visual_diagnostics.py        BS error and IV scatter plots (legacy)
│       ├── mc_path_plot.py              GBM path visualisation (legacy)
│       └── sanity_check.py              Edge-case model tests
│
├── models/                     ML surface models — plugs into evaluation/
│   ├── classical/
│   │   └── polynomial_surface.py   Polynomial regression baseline (Step 8.3)
│   ├── tree/
│   │   └── random_forest_surface.py  Random Forest ensemble (Step 8.3)
│   └── neural/
│       └── mlp_surface.py          Multi-layer perceptron (Step 8.3)
│
├── experiments/                Isolated, reproducible entry points
│   ├── step2_black_scholes/run.py   BS pricing on SPY
│   ├── step3_real_data/run.py       Data collection and cleaning
│   ├── step4_iv/run.py              IV extraction
│   ├── step5_greeks/run.py          Greeks computation
│   ├── step6_binomial/run.py        Binomial pricing
│   ├── step7_monte_carlo/run.py     Monte Carlo pricing
│   └── step8_surface_modeling/run.py  Full surface pipeline
│
├── data/
│   ├── raw/
│   │   ├── spy_options_raw.csv              Raw SPY option chain
│   │   └── options_large/
│   │       └── options_dataset_raw.csv      Raw multi-ticker options
│   └── processed/
│       ├── spy_options_clean.csv            Cleaned SPY options
│       ├── spy_priced_bs.csv                BS priced + errors
│       ├── spy_with_implied_vol.csv         SPY + implied vols
│       ├── spy_priced_binomial.csv          Binomial priced + errors
│       ├── spy_priced_mc.csv                MC priced + errors
│       ├── spy_with_greeks.csv              SPY + analytical Greeks
│       └── options_surface/
│           ├── options_dataset_clean.csv    Cleaned multi-ticker surface
│           ├── options_dataset_features.csv With moneyness features
│           └── options_surface_dataset.csv  Final surface with implied vols
│
├── plots/
│   └── vol_surface/            Saved PNG plots from vol_surface_exploration
│
├── visualization/              Historical plots from earlier project phases
│
├── context/
│   ├── project_context.md      Full project documentation
│   └── repo_structure.md       This file
│
└── notes/                      Step-by-step implementation notes (step0–step7)
```

---

## Data Flow

```
yfinance API
    │
    ├── data_pipeline/ingestion/pull_spy_options.py
    │       → data/raw/spy_options_raw.csv
    │
    └── data_pipeline/ingestion/collect_options_surface.py
            → data/raw/options_large/options_dataset_raw.csv

─── SPY PIPELINE ───────────────────────────────────────────────────────

data/raw/spy_options_raw.csv
    │
    ▼
data_pipeline/cleaning/clean_spy_options.py
    → data/processed/spy_options_clean.csv
    schema: spot, strike, bid, ask, mid, dte, time_to_maturity, moneyness
    │
    ▼
evaluation/pricing/evaluate_pricing_models.price_with_bs()
    → data/processed/spy_priced_bs.csv
    adds: bs_price, bs_error, bs_abs_error, bs_rel_error
    │
    ▼
evaluation/iv/evaluate_iv_models.compute_implied_vols()
    → data/processed/spy_with_implied_vol.csv
    adds: implied_vol
    │
    ├── evaluation/iv/evaluate_iv_models.compute_greeks()
    │       → data/processed/spy_with_greeks.csv
    │       adds: delta, gamma, vega, theta
    │
    ├── evaluation/pricing/evaluate_pricing_models.price_with_binomial()
    │       → data/processed/spy_priced_binomial.csv
    │       adds: binomial_price, binomial_error
    │
    └── evaluation/pricing/evaluate_pricing_models.price_with_mc()
            → data/processed/spy_priced_mc.csv
            adds: mc_price, mc_error

─── SURFACE PIPELINE ────────────────────────────────────────────────────

data/raw/options_large/options_dataset_raw.csv
    │
    ▼
data_pipeline/cleaning/clean_options_dataset.py
    → data/processed/options_surface/options_dataset_clean.csv
    schema: ticker, expiry, strike, bid, ask, mid, spot, time_to_maturity
    │
    ▼
data_pipeline/features/compute_surface_features.py
    → data/processed/options_surface/options_dataset_features.csv
    adds: moneyness, log_moneyness, sqrt_T, moneyness_T_interaction
    │
    ▼
data_pipeline/iv/compute_implied_vols.py
    → data/processed/options_surface/options_surface_dataset.csv
    adds: implied_vol
    │
    ▼
evaluation/diagnostics/vol_surface_exploration.py
    → plots/vol_surface/*.png
    prints: smile structure, term structure, bucket analysis
```

---

## Column Schema

All datasets use this standardised naming:

| Column | Type | Description |
|--------|------|-------------|
| `spot` | float | Underlying price at time of data collection |
| `strike` | float | Option strike price |
| `bid` / `ask` | float | Market bid and ask prices |
| `mid` | float | (bid + ask) / 2 — used as market price proxy |
| `time_to_maturity` | float | Years to expiry: (expiry_date - today) / 365.0 |
| `moneyness` | float | spot / strike (>1 = ITM, <1 = OTM for calls) |
| `implied_vol` | float | Annualised implied volatility (bisection-solved) |
| `ticker` | str | Underlying symbol (surface pipeline only) |

---

## How Each Layer Interacts

```
config/constants.py
    ↓ imported by
    ├── data_pipeline/*   (paths, collection params)
    ├── evaluation/*      (RISK_FREE_RATE, sim params)
    └── experiments/*     (all of the above)

core/pricing/*
    ↓ imported by
    ├── core/iv/implied_vol_solver.py   (BS needed for bisection)
    ├── evaluation/pricing/evaluate_pricing_models.py
    └── evaluation/diagnostics/sanity_check.py

core/iv/implied_vol_solver.py
    ↓ imported by
    ├── evaluation/iv/evaluate_iv_models.py
    └── data_pipeline/iv/compute_implied_vols.py

evaluation/pricing/evaluate_pricing_models.py
    ↓ imported by
    └── experiments/step2, step6, step7

evaluation/iv/evaluate_iv_models.py
    ↓ imported by
    └── experiments/step4, step5

evaluation/diagnostics/error_analysis.py
    ↓ imported by
    └── experiments/step2, step4, step6, step7
```

---

## Running the Full Pipeline

### SPY Single-Asset Pipeline

```bash
# Collect fresh data (optional — existing CSV can be used)
python -m experiments.step3_real_data.run

# Price with Black-Scholes (constant vol baseline)
python -m experiments.step2_black_scholes.run

# Extract implied volatility
python -m experiments.step4_iv.run

# Compute Greeks
python -m experiments.step5_greeks.run

# Price with Binomial tree
python -m experiments.step6_binomial.run

# Price with Monte Carlo
python -m experiments.step7_monte_carlo.run
```

### Multi-Asset Surface Pipeline

```bash
# Collect raw data (optional)
python -m data_pipeline.ingestion.collect_options_surface

# Full cleaning + features + IV pipeline
python -m experiments.step8_surface_modeling.run

# Surface visualisation and diagnostics
python -m evaluation.diagnostics.vol_surface_exploration
```

---

## Where ML Models Will Go (Step 8.3+)

### Adding a new ML model

1. Implement the model class in `models/classical/`, `models/tree/`, or `models/neural/`

2. Expected interface:
   ```python
   model.fit(X_train, y_train)     # X: features DataFrame, y: implied_vol Series
   model.predict(X_test)           # returns: predicted_iv array
   ```

3. Add a training experiment:
   ```
   experiments/step8_surface_modeling/train_<model_name>.py
   ```

4. Add an evaluation script:
   ```
   evaluation/iv/evaluate_<model_name>.py
   ```

5. Feature inputs come from `data/processed/options_surface/options_surface_dataset.csv`:
   - `moneyness`, `log_moneyness`, `time_to_maturity`, `sqrt_T`, `moneyness_T_interaction`
   - `ticker` (encode as categorical)

6. Target: `implied_vol`

The classical bisection solver in `core/iv/implied_vol_solver.py` remains the
ground-truth IV extractor. ML models predict the surface, not extract IV from prices.
