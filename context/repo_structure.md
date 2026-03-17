# Repository Structure

Options pricing research codebase — classical models, IV extraction, and volatility surface analysis.

---

## Directory Tree

```
Options Pricing Model/
│
├── models/                        Core mathematical engines
│   ├── pricing/                   Closed-form and simulation pricers
│   │   ├── black_scholes.py       Black-Scholes call formula (d1, d2, edge cases)
│   │   ├── binomial.py            CRR binomial tree pricer
│   │   ├── monte_carlo.py         Monte Carlo via risk-neutral GBM
│   │   └── greeks.py              Analytical Greeks (delta, gamma, vega, theta)
│   └── iv/                        Implied volatility extraction and modeling
│       ├── implied_vol_solver.py  Bisection solver: market price → implied vol
│       └── iv_models.py           Stub for future ML-based IV surface models
│
├── evaluation/                    Runnable analysis scripts
│   ├── pricing/                   Apply pricers to real data, compute errors
│   │   ├── price_bs.py            Price SPY options with constant historical vol
│   │   ├── price_binomial.py      Price with binomial tree using implied vol
│   │   └── price_mc.py            Price with Monte Carlo using implied vol
│   └── analysis/                  Diagnostics, comparisons, visualizations
│       ├── compute_implied_vol.py Extract IV from SPY market prices
│       ├── compute_greeks.py      Compute Greeks using implied vol
│       ├── sanity_check.py        Edge-case tests for BS and MC models
│       ├── error_analysis.py      Segment BS errors by moneyness and maturity
│       ├── binomial_error_analysis.py  Segment binomial errors, scatter plots
│       ├── implied_vol_analysis.py     IV smile and term structure analysis
│       ├── greek_analysis.py      Greek scatter plots vs moneyness/maturity
│       ├── model_comparison.py    Compare MAE across BS, Binomial, MC
│       ├── visual_diagnostics.py  BS error and IV scatter plots
│       ├── model_comparison_plots.py  Side-by-side error plots (3 models)
│       ├── mc_path_plot.py        Visualize sample GBM paths
│       └── vol_surface_exploration.py  Full multi-asset surface analysis + 3D plots
│
├── processing/                    Data pipeline for multi-asset surface dataset
│   └── data/
│       ├── clean_options_dataset.py   Filter invalid prices, compute time_to_maturity
│       ├── compute_features.py        Add moneyness, log_moneyness, sqrt_T, interactions
│       └── compute_implied_vols.py    Solve IV for all rows, print validation report
│
├── scripts/                       Data collection entry points
│   ├── pull_spy_options.py        Fetch raw SPY option chains via yfinance
│   ├── clean_spy_options.py       Initial SPY cleaning: mid price, moneyness, filters
│   ├── collect_options_surface.py Fetch 5-ticker options surface (SPY, QQQ, AAPL, NVDA, TSLA)
│   └── compute_historical_vol.py  Compute annualized historical vol from SPY price history
│
├── data/
│   ├── raw/
│   │   ├── spy_options_raw.csv              Raw SPY option chain
│   │   └── options_large/
│   │       └── options_dataset_raw.csv      Raw multi-ticker options (5 assets)
│   └── processed/
│       ├── spy_options_clean.csv            Cleaned SPY options
│       ├── spy_priced_bs.csv                SPY priced with constant historical vol
│       ├── spy_with_implied_vol.csv         SPY with extracted implied vols
│       ├── spy_priced_binomial.csv          SPY priced with binomial + IV
│       ├── spy_priced_mc.csv                SPY priced with MC + IV
│       ├── spy_with_greeks.csv              SPY with analytical Greeks
│       └── options_surface/
│           ├── options_dataset_clean.csv    Cleaned multi-ticker surface
│           ├── options_dataset_features.csv With moneyness features
│           └── options_surface_dataset.csv  Final surface with implied vols
│
├── plots/
│   └── vol_surface/               Saved PNG plots from vol_surface_exploration.py
│
├── visualization/                 Historical plots from earlier project phases
│   ├── BS/
│   ├── Binomial/
│   ├── Greeks/
│   ├── IV/
│   ├── Model/
│   └── Monte Carlo/
│
├── context/
│   ├── project_context.md         Full project documentation (models, data, findings)
│   └── repo_structure.md          This file
│
└── notes/                         Step-by-step implementation notes (step0–step7)
```

---

## Purpose of Each Folder

| Folder | Purpose |
|--------|---------|
| `models/pricing/` | Pure mathematical functions — no I/O, no data loading. Import these from anywhere. |
| `models/iv/` | IV extraction (bisection) and future ML IV models. `implied_vol_solver.py` depends only on `models/pricing/black_scholes.py`. |
| `evaluation/pricing/` | Scripts that load processed CSVs, apply a pricer, and save output CSVs with errors. |
| `evaluation/analysis/` | Scripts for diagnostics, visualizations, and cross-model comparisons. |
| `processing/data/` | Multi-step pipeline to clean, featurize, and solve IV for the large multi-asset dataset. |
| `scripts/` | External data collection via yfinance. Run these to refresh raw data files. |
| `data/` | Inputs and outputs for all pipeline stages — never edited manually. |
| `plots/` | Output directory for saved figures from surface exploration. |
| `context/` | Human-readable documentation about the project, models, and findings. |
| `notes/` | Markdown notes written alongside each implementation step. |

---

## Data Flow

```
yfinance API
    │
    ▼
scripts/pull_spy_options.py          → data/raw/spy_options_raw.csv
scripts/collect_options_surface.py   → data/raw/options_large/options_dataset_raw.csv
    │
    ▼
scripts/clean_spy_options.py         → data/processed/spy_options_clean.csv
processing/data/clean_options_dataset.py  → data/processed/options_surface/options_dataset_clean.csv
    │
    ▼
processing/data/compute_features.py  → options_dataset_features.csv
    │
    ▼
evaluation/pricing/price_bs.py       → spy_priced_bs.csv       (constant historical vol)
evaluation/analysis/compute_implied_vol.py → spy_with_implied_vol.csv
processing/data/compute_implied_vols.py    → options_surface_dataset.csv
    │
    ▼
evaluation/pricing/price_binomial.py → spy_priced_binomial.csv (uses implied vol)
evaluation/pricing/price_mc.py       → spy_priced_mc.csv       (uses implied vol)
evaluation/analysis/compute_greeks.py → spy_with_greeks.csv
    │
    ▼
evaluation/analysis/error_analysis.py
evaluation/analysis/model_comparison.py
evaluation/analysis/vol_surface_exploration.py   → plots/vol_surface/*.png
```

---

## Where Models Live

All pricing models are pure functions in `models/pricing/`:

| File | Function | Description |
|------|----------|-------------|
| `black_scholes.py` | `black_scholes_call(S, K, T, r, sigma)` | Closed-form European call |
| `binomial.py` | `binomial_call_price(S, K, T, r, sigma, steps)` | CRR binomial tree |
| `monte_carlo.py` | `monte_carlo_call_price(S, K, T, r, sigma, n_sims, seed)` | GBM simulation |
| `greeks.py` | `delta_call`, `gamma_call`, `vega_call`, `theta_call` | Analytical sensitivities |

The IV solver lives in `models/iv/implied_vol_solver.py`:
- `implied_vol_bisection(market_price, S, K, T, r)` — finds σ such that BS(σ) = market_price

---

## Where Evaluation Happens

**Single-asset pipeline (SPY):**

Run scripts in this order from the project root:
```bash
python -m evaluation.pricing.price_bs
python -m evaluation.analysis.compute_implied_vol
python -m evaluation.pricing.price_binomial
python -m evaluation.pricing.price_mc
python -m evaluation.analysis.compute_greeks
python -m evaluation.analysis.error_analysis
python -m evaluation.analysis.model_comparison
```

**Multi-asset surface pipeline:**

```bash
python -m processing.data.clean_options_dataset
python -m processing.data.compute_features
python -m processing.data.compute_implied_vols
python -m evaluation.analysis.vol_surface_exploration
```

---

## Where ML Models Will Go (Step 8.3+)

New ML-based IV surface models belong in `models/iv/iv_models.py`.

The expected interface:
- **Input:** features from `options_surface_dataset.csv` — moneyness, log_moneyness, time_to_maturity, sqrt_T, moneyness_T_interaction, ticker
- **Output:** predicted implied volatility

Training scripts should live in `evaluation/analysis/` (e.g., `train_iv_model.py`, `evaluate_iv_model.py`).

Fitted model artifacts (weights, checkpoints) can go in a new `models/iv/checkpoints/` directory.

The classical bisection solver in `implied_vol_solver.py` remains the ground-truth IV extractor and should not be replaced — ML models predict surface IV, while the bisection solver extracts it from a single observed price.
