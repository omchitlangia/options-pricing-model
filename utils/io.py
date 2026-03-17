"""
io.py
------
Reusable I/O utilities for loading and saving datasets.

All pipeline scripts should use these functions instead of calling
pd.read_csv / df.to_csv directly, so that logging and error handling
are centralised.
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame. Prints row count on load."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV. Prints row count on save."""
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")
