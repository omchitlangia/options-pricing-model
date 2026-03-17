"""
iv_models.py
-------------
Placeholder for future implied volatility models (Step 8.3+).

Planned additions:
  - SVI (Stochastic Volatility Inspired) parametrization
  - SABR model implied vol approximation
  - Neural network / ML-based IV surface fitting

The implied_vol_solver.py in this package provides the bisection-based
IV extractor used throughout the current pipeline. Models here will instead
*predict* IV from market features (moneyness, maturity, ticker, etc.)
rather than back out IV from a single market price.

Expected input format (from options_surface_dataset.csv):
  moneyness, log_moneyness, time_to_maturity, sqrt_T,
  moneyness_T_interaction, ticker

Expected output:
  predicted_iv  (float, annualized)
"""

# TODO (Step 8.3): implement ML-based IV surface models
