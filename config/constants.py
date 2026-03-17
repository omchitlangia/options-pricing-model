"""
constants.py
-------------
Single source of truth for all project-wide parameters.

All pricing scripts, data pipeline steps, and experiment runners
must import constants from here rather than hardcoding values inline.
"""

# -----------------------------------------------------------------------
# Market / Pricing Parameters
# -----------------------------------------------------------------------

RISK_FREE_RATE: float = 0.036
"""3-month US T-bill rate used consistently across all pricing computations."""

HISTORICAL_SIGMA_SPY: float = 0.1132
"""
Annualized historical volatility computed from 90-day SPY log returns (Step 3.3).
Used in Black-Scholes pricing as the constant-vol baseline.
"""

# -----------------------------------------------------------------------
# Simulation Parameters
# -----------------------------------------------------------------------

MONTE_CARLO_N_SIMS: int = 50_000
MONTE_CARLO_SEED: int = 42
BINOMIAL_STEPS: int = 200

# -----------------------------------------------------------------------
# Data Collection Parameters
# -----------------------------------------------------------------------

SPY_MIN_DTE: int = 7
SPY_MAX_DTE: int = 45
SPY_MONEYNESS_LOW: float = 0.85
SPY_MONEYNESS_HIGH: float = 1.15

SURFACE_TICKERS: list = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
SURFACE_MAX_EXPIRIES: int = 12
SURFACE_MAX_SPREAD_RATIO: float = 0.50

# -----------------------------------------------------------------------
# Plot Parameters
# -----------------------------------------------------------------------

SURFACE_MONEYNESS_LO: float = 0.70
SURFACE_MONEYNESS_HI: float = 1.40
SURFACE_IV_MAX_PLOT: float = 2.50

TICKER_COLORS: dict = {
    "SPY":  "steelblue",
    "QQQ":  "darkorange",
    "AAPL": "green",
    "NVDA": "red",
    "TSLA": "purple",
}

# -----------------------------------------------------------------------
# Data Paths — Raw
# -----------------------------------------------------------------------

RAW_SPY_OPTIONS: str         = "data/raw/spy_options_raw.csv"
RAW_OPTIONS_SURFACE: str     = "data/raw/options_large/options_dataset_raw.csv"

# -----------------------------------------------------------------------
# Data Paths — Processed SPY pipeline
# -----------------------------------------------------------------------

PROCESSED_SPY_CLEAN: str     = "data/processed/spy_options_clean.csv"
PROCESSED_SPY_BS: str        = "data/processed/spy_priced_bs.csv"
PROCESSED_SPY_IV: str        = "data/processed/spy_with_implied_vol.csv"
PROCESSED_SPY_BINOMIAL: str  = "data/processed/spy_priced_binomial.csv"
PROCESSED_SPY_MC: str        = "data/processed/spy_priced_mc.csv"
PROCESSED_SPY_GREEKS: str    = "data/processed/spy_with_greeks.csv"

# -----------------------------------------------------------------------
# Data Paths — Processed surface pipeline
# -----------------------------------------------------------------------

PROCESSED_SURFACE_CLEAN: str    = "data/processed/options_surface/options_dataset_clean.csv"
PROCESSED_SURFACE_FEATURES: str = "data/processed/options_surface/options_dataset_features.csv"
PROCESSED_SURFACE_DATASET: str  = "data/processed/options_surface/options_surface_dataset.csv"

# -----------------------------------------------------------------------
# Output Paths — Plots
# -----------------------------------------------------------------------

PLOT_DIR_VOL_SURFACE: str = "plots/vol_surface"
