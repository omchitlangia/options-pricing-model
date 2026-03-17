import math
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def black_scholes_call(S, K, T, r, sigma):
    if sigma == 0:
        return max(S - K * math.exp(-r * T), 0)

    if T == 0:
        return max(S - K, 0)

    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    return S * norm.cdf(D1) - K * math.exp(-r * T) * norm.cdf(D2)
