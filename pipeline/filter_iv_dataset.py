"""
filter_iv_dataset.py

Financial data curation pipeline for implied volatility surface modeling.

Design principle: distinguish between data errors / microstructure noise (REMOVE)
and legitimate market structure (KEEP). The volatility smile, skew, and wing
elevations are real features — aggressive trimming would artificially flatten
the surface and corrupt any downstream model.

Filter sequence:
    1. Sanity checks      — catch structurally invalid quotes
    2. Liquidity          — remove quotes where bid/ask spread signals no real market
    3. Maturity           — remove near-expiry options with unstable IV
    4. Moneyness          — restrict to the surface modeling range; flag extreme wings
    5. Implied vol        — remove numerical solver failures only
    6. Weighting          — soft alternative to hard removal; preserves structure
"""

import numpy as np
import pandas as pd

INPUT_PATH  = "data/processed/options_surface/options_surface_dataset.csv"
OUTPUT_PATH = "data/processed/options_surface/options_surface_filtered.csv"


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Recompute mid from the live bid/ask rather than trusting the stored value.
    # The dataset mid column was computed upstream, but we recalculate here so
    # all downstream filters are consistent with the actual spread.
    df["mid"] = (df["bid"] + df["ask"]) / 2

    return df


# ─────────────────────────────────────────────────────────────
# STEP 1 — SANITY FILTERS
# ─────────────────────────────────────────────────────────────
# These catch structurally broken quotes that cannot represent real market data.
# A negative bid, an ask below bid, or a zero mid price means the quote is either
# a data feed error or a stale placeholder.  Nothing useful can be inferred from
# such rows and they must be removed before any numerical operation.

def apply_sanity_filters(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)

    # Negative or zero bid/ask: not a real two-sided market
    mask = (df["bid"] > 0) & (df["ask"] > 0)

    # Crossed market: bid above ask is a feed error, not a real spread
    mask &= df["bid"] <= df["ask"]

    # Zero or negative mid: the option has no recoverable price signal
    mask &= df["mid"] > 0

    # Very small mid (< $0.01): the option price is below normal tick resolution.
    # IV extracted from prices this small is dominated by rounding, not volatility.
    # This is a data-quality floor, not a financial judgement about the option.
    mask &= df["mid"] >= 0.01

    df = df[mask].copy()
    _report("Sanity filters", n_in, len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — LIQUIDITY FILTERS
# ─────────────────────────────────────────────────────────────
# The relative bid/ask spread measures how wide the market maker quotes relative
# to the option's price.  A very wide spread (> 1.0×  mid) means the quoted
# range is wider than the price itself — essentially no real two-sided market.
#
# We do NOT remove options just because they have a moderate spread or low volume.
# A deep-OTM option with low volume can still carry important smile information.
# The only hard removals are: spread > 1.0, OR zero volume AND zero open interest
# AND a high spread (all three together reliably signal a stale, untradeable quote).
#
# Flagging (wide_spread column) is used instead of removal for spread in (0.3, 1.0].
# Downstream models can use weights to down-weight these rows rather than drop them.

def apply_liquidity_filters(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)

    df["relative_spread"] = (df["ask"] - df["bid"]) / df["mid"]

    # Hard remove: spread > 1.0 — no real market exists at this width
    mask = df["relative_spread"] <= 1.0

    # Hard remove: zero volume AND zero open interest AND spread > 0.3
    # All three conditions together indicate a completely stale, untraded quote.
    # Each condition alone is not sufficient: illiquid options may still have OI,
    # and a spread > 0.3 with positive OI may be a legitimate deep-wing quote.
    vol  = df["volume"].fillna(0)
    oi   = df["openInterest"].fillna(0)
    dead_quote = (vol == 0) & (oi == 0) & (df["relative_spread"] > 0.3)
    mask &= ~dead_quote

    # Soft flag: spread > 0.3 signals elevated uncertainty but NOT removal.
    # These rows will receive lower weights in step 6.
    df["wide_spread_flag"] = (df["relative_spread"] > 0.3).astype(int)

    df = df[mask].copy()
    _report("Liquidity filters", n_in, len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — MATURITY FILTERS
# ─────────────────────────────────────────────────────────────
# Near-expiry options (T < 1 day ≈ 0.003 years) behave pathologically:
#   - Gamma explodes near expiry, making IV extremely sensitive to tiny price moves
#   - A $0.01 bid/ask tick can imply IV differences of tens of percentage points
#   - The Black-Scholes IV solver frequently fails or returns unstable values
# These options are structurally unsuitable for surface modeling and are removed.
#
# Options with T in [0.003, 0.02) (~1–7 days) are kept but flagged.  They are
# valid market data and inform the very short end of the term structure, but
# downstream models should weight them carefully or treat them separately.

def apply_maturity_filters(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)

    # Remove: less than ~1 trading day — IV is numerically unstable at this horizon
    mask = df["time_to_maturity"] >= 0.003

    # Soft flag: short-dated (1–7 days) — valid data, but handle with care
    df["short_maturity_flag"] = (df["time_to_maturity"] < 0.02).astype(int)

    df = df[mask].copy()
    _report("Maturity filters", n_in, len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — MONEYNESS FILTERS
# ─────────────────────────────────────────────────────────────
# The moneyness range [0.7, 1.3] is the financial modeling range for this dataset.
# Options outside this range are not removed because they are "extreme" — many are
# legitimate deep-ITM options from TSLA and AAPL.  They are removed because:
#   - Moneyness > 1.3: deep-ITM calls are far from the smile region; their IV is
#     driven by intrinsic value and is not informative about market risk sentiment.
#     Additionally, 85 rows in this region have IV capped at 3.0 (solver failure).
#   - Moneyness < 0.7: deep-OTM calls with near-zero prices; IV is unstable.
#
# DO NOT remove options just because they are in the tails [0.7, 0.85] or [1.15, 1.3].
# The volatility smile exists precisely because these wings carry elevated IV.
# Removing them would erase the smile curvature that defines the surface structure.
#
# Wing flags mark rows in [0.7, 0.85] and [1.15, 1.3] for diagnostic use.
# These rows should receive normal treatment in modeling; the flag is informational.

def apply_moneyness_filters(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)

    # Surface modeling range: keep [0.7, 1.3]
    mask = (df["moneyness"] > 0.7) & (df["moneyness"] < 1.3)

    # Soft flag: extreme wings within the kept range
    # These are legitimate market data and must be retained for smile modeling.
    # The flag allows downstream analysis to inspect tail behavior separately.
    df["wing_flag"] = (
        (df["moneyness"] < 0.85) | (df["moneyness"] > 1.15)
    ).astype(int)

    df = df[mask].copy()
    _report("Moneyness filters", n_in, len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — IMPLIED VOLATILITY FILTERS
# ─────────────────────────────────────────────────────────────
# The IV filter must be conservative.  High IV in the wings (0.5–1.5) is a real
# market phenomenon driven by:
#   - Demand for tail protection (puts) or levered upside (calls)
#   - Jump risk premium in single-stock options (TSLA, NVDA)
#   - Term structure effects compressing near-dated IV higher
# Removing or winsorizing these rows would suppress the very signal we want to model.
#
# Hard removal bounds:
#   IV < 0.01: below any reasonable market vol; almost certainly a solver failure
#              where the option was priced at or below intrinsic value.
#   IV >= 3.0: the bisection solver used upstream has a hard cap of 3.0.  Rows at
#              exactly 3.0 are not true market IVs — the solver hit its ceiling and
#              could not converge. These rows are solver artifacts, not market data.
#              (At this dataset's moneyness bounds, these rows are already removed
#               by the moneyness filter, but this check ensures correctness regardless.)
#
# Winsorization at the 99th percentile is applied as a SOFT cap for modeling
# stability.  The original IV is preserved in implied_vol_raw so the true market
# signal is never destroyed.

def apply_iv_filters(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)

    # Remove solver failures at both ends
    mask = (df["implied_vol"] >= 0.01) & (df["implied_vol"] < 3.0)
    df = df[mask].copy()

    # Preserve original IV before any capping
    df["implied_vol_raw"] = df["implied_vol"].copy()

    # Soft winsorization at 99th percentile.
    # This prevents extreme solver near-cap values from dominating model loss
    # during training, while keeping the row in the dataset with its true IV
    # available in implied_vol_raw.
    p99 = df["implied_vol"].quantile(0.99)
    df["implied_vol"] = df["implied_vol"].clip(upper=p99)

    _report("IV filters", n_in, len(df))
    print(f"  IV winsorization cap (99th pct): {p99:.4f}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 — WEIGHTING SCHEME
# ─────────────────────────────────────────────────────────────
# Instead of removing borderline rows, we assign weights that reflect data quality.
# This allows models to use all retained data while naturally discounting noisy rows.
#
# Base weight — inverse of relative spread:
#   A tight spread means the market maker is confident in the price; the quote
#   carries more information.  weight = 1 / (1 + spread) maps [0, ∞) to (0, 1].
#
# ATM boost — exponential decay away from ATM:
#   ATM options anchor the level of the vol surface.  Their prices are the most
#   reliably observed and their IV is the most stable numerically.  We give them
#   higher weight so the surface level is well-identified in regression models.
#   weight *= exp(-|log_moneyness|)  peaks at 1.0 for ATM and decays toward wings.
#   This does NOT remove wings — it only means ATM quotes dominate level fitting
#   while wing quotes still inform the smile shape.
#
# The final weight is normalized to [0, 1] for interpretability.

def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    # Base: reward tight spreads
    w = 1.0 / (1.0 + df["relative_spread"])

    # ATM boost: discount wings gently, not aggressively
    w *= np.exp(-np.abs(df["log_moneyness"]))

    # Normalize to [0, 1]
    df["weight"] = w / w.max()

    return df


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    n_raw   = len(df_raw)
    n_clean = len(df_clean)
    pct_removed = 100 * (1 - n_clean / n_raw)

    print("\n" + "═" * 50)
    print("FILTER SUMMARY")
    print("═" * 50)
    print(f"  Original rows : {n_raw:>6}")
    print(f"  Filtered rows : {n_clean:>6}")
    print(f"  Removed       : {n_raw - n_clean:>6}  ({pct_removed:.1f}%)")

    print("\n── Implied Volatility (filtered) ──")
    iv = df_clean["implied_vol"]
    print(f"  mean   : {iv.mean():.4f}")
    print(f"  median : {iv.median():.4f}")
    print(f"  std    : {iv.std():.4f}")
    print(f"  min    : {iv.min():.4f}")
    print(f"  max    : {iv.max():.4f}")

    print("\n── Relative Spread (filtered) ──")
    sp = df_clean["relative_spread"]
    print(f"  mean   : {sp.mean():.4f}")
    print(f"  median : {sp.median():.4f}")
    print(f"  max    : {sp.max():.4f}")
    print(f"  flagged (> 0.3): {df_clean['wide_spread_flag'].sum()} rows")

    print("\n── Flags ──")
    print(f"  short_maturity_flag : {df_clean['short_maturity_flag'].sum()} rows  (T < 0.02)")
    print(f"  wing_flag           : {df_clean['wing_flag'].sum()} rows  (moneyness < 0.85 or > 1.15)")

    print("\n── Rows per ticker ──")
    print(df_clean["ticker"].value_counts().to_string())

    print("\n── Weight distribution ──")
    w = df_clean["weight"]
    print(f"  mean   : {w.mean():.4f}")
    print(f"  median : {w.median():.4f}")
    print(f"  min    : {w.min():.4f}")
    print("═" * 50)


def _report(step: str, n_before: int, n_after: int) -> None:
    removed = n_before - n_after
    print(f"  [{step}]  {n_before} → {n_after}  (removed {removed})")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    df = load_data(INPUT_PATH)
    df_raw = df.copy()

    print("\nApplying filters:")
    df = apply_sanity_filters(df)
    df = apply_liquidity_filters(df)
    df = apply_maturity_filters(df)
    df = apply_moneyness_filters(df)
    df = apply_iv_filters(df)
    df = compute_weights(df)

    print_summary(df_raw, df)
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
