"""
evaluate_pricing_models.py
---------------------------
Reusable functions for pricing SPY options with all three models
(Black-Scholes, Binomial, Monte Carlo) and computing pricing errors.

Functions here accept DataFrames as input and return enriched DataFrames.
They do NOT load or save files — that is the responsibility of the caller
(typically an experiment script in experiments/).

Column schema expected:
  spot, strike, time_to_maturity, mid, implied_vol (for binomial / MC)
"""

import pandas as pd

from core.black_scholes import black_scholes_call
from core.binomial import binomial_call_price
from core.monte_carlo import monte_carlo_call_price
from config.constants import (
    RISK_FREE_RATE,
    HISTORICAL_SIGMA_SPY,
    BINOMIAL_STEPS,
    MONTE_CARLO_N_SIMS,
    MONTE_CARLO_SEED,
)


def price_with_bs(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    sigma: float = HISTORICAL_SIGMA_SPY,
) -> pd.DataFrame:
    """
    Price every option in df using Black-Scholes with a constant sigma.
    Adds columns: bs_price, bs_error, bs_abs_error, bs_rel_error.
    """
    df = df.copy()
    df["bs_price"] = df.apply(
        lambda x: black_scholes_call(
            S=x["spot"],
            K=x["strike"],
            T=x["time_to_maturity"],
            r=risk_free_rate,
            sigma=sigma,
        ),
        axis=1,
    )
    df["bs_error"]     = df["bs_price"] - df["mid"]
    df["bs_abs_error"] = df["bs_error"].abs()
    df["bs_rel_error"] = df["bs_error"] / df["mid"]
    return df


def price_with_binomial(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    steps: int = BINOMIAL_STEPS,
) -> pd.DataFrame:
    """
    Price every option using the CRR binomial tree with per-row implied_vol.
    Adds columns: binomial_price, binomial_error.
    Expects column: implied_vol.
    """
    df = df.copy()
    prices, errors = [], []
    for _, row in df.iterrows():
        p = binomial_call_price(
            S=row["spot"],
            K=row["strike"],
            T=row["time_to_maturity"],
            r=risk_free_rate,
            sigma=row["implied_vol"],
            steps=steps,
        )
        prices.append(p)
        errors.append(p - row["mid"])
    df["binomial_price"] = prices
    df["binomial_error"] = errors
    return df


def price_with_mc(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    n_sims: int = MONTE_CARLO_N_SIMS,
    seed: int = MONTE_CARLO_SEED,
) -> pd.DataFrame:
    """
    Price every option using Monte Carlo GBM with per-row implied_vol.
    Adds columns: mc_price, mc_error.
    Expects column: implied_vol.
    """
    df = df.copy()
    prices, errors = [], []
    for _, row in df.iterrows():
        p = monte_carlo_call_price(
            S=row["spot"],
            K=row["strike"],
            T=row["time_to_maturity"],
            r=risk_free_rate,
            sigma=row["implied_vol"],
            n_sims=n_sims,
            seed=seed,
        )
        prices.append(p)
        errors.append(p - row["mid"])
    df["mc_price"] = prices
    df["mc_error"] = errors
    return df


def print_pricing_summary(df: pd.DataFrame) -> None:
    """Print MAE for whichever pricing columns are present."""
    print(f"\nPricing Summary ({len(df)} options)")
    if "bs_abs_error"   in df.columns:
        print(f"  BS MAE       : {df['bs_abs_error'].mean():.4f}")
    if "binomial_error" in df.columns:
        print(f"  Binomial MAE : {df['binomial_error'].abs().mean():.4f}")
    if "mc_error"       in df.columns:
        print(f"  MC MAE       : {df['mc_error'].abs().mean():.4f}")
