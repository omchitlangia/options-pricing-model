"""
experiments/step8_surface_modeling/run.py
-------------------------------------------
Step 8 — Volatility Surface Modeling

Entry point for the multi-asset surface pipeline and exploration.
Runs the full surface pipeline and produces diagnostic visualisations.

Sub-steps:
  8.1  Collect multi-ticker options data  →  pipeline/
  8.2  Volatility surface exploration     →  evaluation/vol_surface_exploration.py
  8.3  ML surface models (upcoming)       →  models/

Run from the project root:
    python -m experiments.step8_surface_modeling.run
"""

from utils.logging import print_header
from pipeline.clean_options_dataset import main as run_cleaning
from pipeline.compute_surface_features import main as run_features
from pipeline.compute_implied_vols import main as run_iv


def main() -> None:
    print_header("Step 8 — Volatility Surface Pipeline")

    print("\n[1/3] Cleaning raw surface dataset...")
    run_cleaning()

    print("\n[2/3] Computing surface features...")
    run_features()

    print("\n[3/3] Solving implied volatilities...")
    run_iv()

    print("\n" + "=" * 60)
    print("Surface pipeline complete.")
    print("Run vol_surface_exploration for diagnostics and plots:")
    print("  python -m evaluation.vol_surface_exploration")
    print("=" * 60)


if __name__ == "__main__":
    main()
