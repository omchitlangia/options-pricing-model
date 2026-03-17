"""
evaluate_iv_models.py
----------------------
Reusable functions for implied volatility computation and Greeks
calculation on the SPY single-asset pipeline.

Column schema expected:
  spot, strike, time_to_maturity, mid   (for IV)
  spot, strike, time_to_maturity, implied_vol  (for Greeks)
"""

import pandas as pd

from core.implied_vol import implied_vol_bisection
from core.greeks import delta_call, gamma_call, vega_call, theta_call
from config.constants import RISK_FREE_RATE


def compute_implied_vols(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Extract implied volatility for each option using the bisection solver.
    Rows where the solver fails (price below intrinsic) are dropped.
    Adds column: implied_vol.
    """
    df = df.copy()
    df["implied_vol"] = df.apply(
        lambda x: implied_vol_bisection(
            market_price=x["mid"],
            S=x["spot"],
            K=x["strike"],
            T=x["time_to_maturity"],
            r=risk_free_rate,
        ),
        axis=1,
    )
    n_failed = df["implied_vol"].isna().sum()
    df = df.dropna(subset=["implied_vol"])
    if n_failed:
        print(f"  IV solver failed on {n_failed} rows (dropped)")
    return df


def compute_greeks(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Compute analytical Black-Scholes Greeks using per-row implied_vol.
    Adds columns: delta, gamma, vega, theta.
    Expects column: implied_vol.
    """
    df = df.copy()
    df["delta"] = df.apply(
        lambda x: delta_call(x["spot"], x["strike"], x["time_to_maturity"], risk_free_rate, x["implied_vol"]),
        axis=1,
    )
    df["gamma"] = df.apply(
        lambda x: gamma_call(x["spot"], x["strike"], x["time_to_maturity"], risk_free_rate, x["implied_vol"]),
        axis=1,
    )
    df["vega"] = df.apply(
        lambda x: vega_call(x["spot"], x["strike"], x["time_to_maturity"], risk_free_rate, x["implied_vol"]),
        axis=1,
    )
    df["theta"] = df.apply(
        lambda x: theta_call(x["spot"], x["strike"], x["time_to_maturity"], risk_free_rate, x["implied_vol"]),
        axis=1,
    )
    return df


def print_iv_summary(df: pd.DataFrame) -> None:
    iv = df["implied_vol"]
    print(f"\nImplied Volatility Summary ({len(df)} options)")
    print(f"  mean : {iv.mean():.4f}")
    print(f"  std  : {iv.std():.4f}")
    print(f"  min  : {iv.min():.4f}")
    print(f"  max  : {iv.max():.4f}")


def print_greeks_summary(df: pd.DataFrame) -> None:
    print(f"\nGreeks Summary ({len(df)} options)")
    print(df[["delta", "gamma", "vega", "theta"]].describe().round(4).to_string())
