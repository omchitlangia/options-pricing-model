import math
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def delta_call(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))


def gamma_call(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * math.sqrt(T))


def vega_call(S, K, T, r, sigma):
    # Vega per 1.0 change in volatility (not 1%)
    return S * norm.pdf(d1(S, K, T, r, sigma)) * math.sqrt(T)


def theta_call(S, K, T, r, sigma):
    term1 = -(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * math.sqrt(T))
    term2 = -r * K * math.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))
    return term1 + term2
