"""
vol_surface_exploration.py
--------------------------
Step 8.2 — Volatility Surface Exploration

Loads the multi-asset options surface dataset and produces:
  1. Full dataset summary statistics
  2. Scatter: implied volatility vs moneyness  (volatility smile)
  3. Scatter: implied volatility vs time to maturity  (term structure)
  4. 3D scatter: implied volatility surface over (moneyness × maturity)
  5. Smoothed triangulated surface using plot_trisurf
  6. Diagnostic bucket analysis — mean IV by moneyness and maturity bucket

All plots are saved to:  plots/vol_surface/

Run from the project root:
    python -m evaluation.vol_surface_exploration
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection

from config.constants import (
    PROCESSED_SURFACE_DATASET,
    PLOT_DIR_VOL_SURFACE,
    SURFACE_MONEYNESS_LO,
    SURFACE_MONEYNESS_HI,
    SURFACE_IV_MAX_PLOT,
    TICKER_COLORS,
)

DATA_PATH    = PROCESSED_SURFACE_DATASET
PLOT_DIR     = PLOT_DIR_VOL_SURFACE
MONEYNESS_LO = SURFACE_MONEYNESS_LO
MONEYNESS_HI = SURFACE_MONEYNESS_HI
IV_MAX_PLOT  = SURFACE_IV_MAX_PLOT


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY  (full dataset, unfiltered)")
    print("=" * 60)
    print(f"Total rows            : {len(df)}")
    print(f"Tickers               : {sorted(df['ticker'].unique())}")

    print(f"\nRows per ticker:")
    for ticker, count in df.groupby("ticker").size().items():
        print(f"  {ticker}: {count}")

    print(f"\nUnique expiries per ticker:")
    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker]
        n_exp = sub["expiry"].nunique()
        t_min = sub["time_to_maturity"].min()
        t_max = sub["time_to_maturity"].max()
        print(f"  {ticker}: {n_exp} expiries  T ∈ [{t_min:.4f}, {t_max:.4f}] years")

    m_min = df["moneyness"].min()
    m_max = df["moneyness"].max()
    t_min = df["time_to_maturity"].min()
    t_max = df["time_to_maturity"].max()
    print(f"\nMoneyness range       : {m_min:.4f} — {m_max:.4f}")
    print(f"Time to maturity range: {t_min:.4f} — {t_max:.4f} years")

    iv = df["implied_vol"]
    print(f"\nImplied Volatility:")
    print(f"  mean : {iv.mean():.4f}")
    print(f"  std  : {iv.std():.4f}")
    print(f"  min  : {iv.min():.4f}")
    print(f"  max  : {iv.max():.4f}")

    n_capped = (iv >= 3.0).sum()
    n_extreme = (df["moneyness"] > MONEYNESS_HI).sum() + \
                (df["moneyness"] < MONEYNESS_LO).sum()
    print(f"\nOutlier notes:")
    print(f"  IV = 3.0 (solver cap)        : {n_capped} rows")
    print(f"  Moneyness outside [{MONEYNESS_LO}, {MONEYNESS_HI}]: {n_extreme} rows")
    print("=" * 60)


def filter_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["moneyness"].between(MONEYNESS_LO, MONEYNESS_HI) &
        (df["implied_vol"] < IV_MAX_PLOT)
    )
    df_plot = df[mask].copy()
    print(
        f"\nPlot dataset: {len(df_plot)} rows  "
        f"(moneyness ∈ [{MONEYNESS_LO}, {MONEYNESS_HI}], IV < {IV_MAX_PLOT})"
    )
    return df_plot


def plot_iv_vs_moneyness(df: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker]
        ax.scatter(
            sub["moneyness"], sub["implied_vol"],
            label=ticker, color=TICKER_COLORS.get(ticker, "gray"),
            alpha=0.45, s=16, linewidths=0
        )
    ax.axvline(x=1.0, color="black", linewidth=0.9, linestyle="--", label="ATM  (moneyness = 1)")
    ax.set_xlabel("Moneyness  (spot / strike)", fontsize=12)
    ax.set_ylabel("Implied Volatility", fontsize=12)
    ax.set_title("Implied Volatility vs Moneyness — Volatility Smile", fontsize=13, fontweight="bold")
    ax.legend(title="Ticker", fontsize=9, title_fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_iv_vs_maturity(df: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker]
        ax.scatter(
            sub["time_to_maturity"], sub["implied_vol"],
            label=ticker, color=TICKER_COLORS.get(ticker, "gray"),
            alpha=0.45, s=16, linewidths=0
        )
    ax.set_xlabel("Time to Maturity (years)", fontsize=12)
    ax.set_ylabel("Implied Volatility", fontsize=12)
    ax.set_title("Implied Volatility vs Time to Maturity — Term Structure", fontsize=13, fontweight="bold")
    ax.legend(title="Ticker", fontsize=9, title_fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_iv_surface_3d(df: pd.DataFrame, save_path: str) -> None:
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker]
        ax.scatter(
            sub["moneyness"].values, sub["time_to_maturity"].values, sub["implied_vol"].values,
            label=ticker, color=TICKER_COLORS.get(ticker, "gray"),
            alpha=0.55, s=14, linewidths=0
        )
    ax.set_xlabel("Moneyness  (S / K)", fontsize=10, labelpad=10)
    ax.set_ylabel("Time to Maturity (years)", fontsize=10, labelpad=10)
    ax.set_zlabel("Implied Volatility", fontsize=10, labelpad=10)
    ax.set_title("Implied Volatility Surface", fontsize=13, fontweight="bold")
    ax.legend(title="Ticker", fontsize=8, title_fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_iv_surface_smoothed(df: pd.DataFrame, save_path: str) -> None:
    if len(df) < 50:
        print("  Smoothed surface skipped: insufficient data points.")
        return
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        df["moneyness"].values, df["time_to_maturity"].values, df["implied_vol"].values,
        cmap="viridis", alpha=0.80, linewidth=0.15, edgecolor="none"
    )
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="Implied Volatility")
    ax.set_xlabel("Moneyness  (S / K)", fontsize=10, labelpad=10)
    ax.set_ylabel("Time to Maturity (years)", fontsize=10, labelpad=10)
    ax.set_zlabel("Implied Volatility", fontsize=10, labelpad=10)
    ax.set_title("Implied Volatility Surface — Smoothed (all tickers)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


def print_bucket_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DIAGNOSTIC BUCKET ANALYSIS")
    print("=" * 60)

    mono_bins   = [0.80, 0.90, 1.00, 1.10, 1.20]
    mono_labels = ["0.80–0.90", "0.90–1.00", "1.00–1.10", "1.10–1.20"]
    mono_bucket = pd.cut(df["moneyness"], bins=mono_bins, labels=mono_labels, include_lowest=True)

    print("\nMean IV by moneyness bucket (all tickers combined):")
    agg = df.groupby(mono_bucket, observed=True)["implied_vol"].agg(["mean", "count"])
    for bucket, row in agg.iterrows():
        bar = "█" * int(row["mean"] * 20)
        print(f"  {bucket}: {row['mean']:.4f}  (n={int(row['count']):4d})  {bar}")

    print("\nMean IV by moneyness bucket — per ticker:")
    df["_mono_bucket"] = mono_bucket
    pivot_mono = (
        df.groupby(["ticker", "_mono_bucket"], observed=True)["implied_vol"]
          .mean()
          .unstack(level=1)
          .round(4)
    )
    print(pivot_mono.to_string())
    df.drop(columns=["_mono_bucket"], inplace=True)

    mat_bins   = [0.00, 0.05, 0.15, 0.30, 9.00]
    mat_labels = ["0–0.05", "0.05–0.15", "0.15–0.30", "0.30+"]
    mat_bucket = pd.cut(df["time_to_maturity"], bins=mat_bins, labels=mat_labels, include_lowest=True)

    print("\nMean IV by maturity bucket (all tickers combined):")
    agg_mat = df.groupby(mat_bucket, observed=True)["implied_vol"].agg(["mean", "count"])
    for bucket, row in agg_mat.iterrows():
        bar = "█" * int(row["mean"] * 20)
        print(f"  {bucket:10s}: {row['mean']:.4f}  (n={int(row['count']):4d})  {bar}")

    print("\nMean IV by maturity bucket — per ticker:")
    df["_mat_bucket"] = mat_bucket
    pivot_mat = (
        df.groupby(["ticker", "_mat_bucket"], observed=True)["implied_vol"]
          .mean()
          .unstack(level=1)
          .round(4)
    )
    print(pivot_mat.to_string())
    df.drop(columns=["_mat_bucket"], inplace=True)

    print("=" * 60)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = load_dataset(DATA_PATH)
    print_summary(df)

    df_plot = filter_for_plots(df)

    plot_iv_vs_moneyness(df_plot, save_path=os.path.join(PLOT_DIR, "iv_vs_moneyness.png"))
    plot_iv_vs_maturity(df_plot, save_path=os.path.join(PLOT_DIR, "iv_vs_maturity.png"))
    plot_iv_surface_3d(df_plot, save_path=os.path.join(PLOT_DIR, "iv_surface_3d.png"))
    plot_iv_surface_smoothed(df_plot, save_path=os.path.join(PLOT_DIR, "iv_surface_smoothed.png"))
    print_bucket_analysis(df_plot)

    print(f"\nAll plots saved to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
