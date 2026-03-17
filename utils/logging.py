"""
logging.py
-----------
Lightweight console logging utilities for structured output.

Keeps print formatting consistent across all pipeline scripts
and experiment runners without introducing a logging framework.
"""


def print_header(title: str, width: int = 60) -> None:
    """Print a top-level section header."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_section(title: str) -> None:
    """Print a sub-section title with an underline."""
    print(f"\n{title}")
    print("-" * len(title))
