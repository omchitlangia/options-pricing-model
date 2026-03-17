"""
distributions.py
-----------------
Centralised normal distribution utilities.

Black-Scholes and Greeks both depend on the standard normal CDF and PDF.
Importing from here ensures a single dependency on scipy.stats across
the entire core layer.
"""

from scipy.stats import norm


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function: Φ(x)."""
    return norm.cdf(x)


def normal_pdf(x: float) -> float:
    """Standard normal probability density function: φ(x)."""
    return norm.pdf(x)
