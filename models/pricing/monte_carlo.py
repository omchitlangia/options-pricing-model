import numpy as np

def monte_carlo_call_price(
    S,
    K,
    T,
    r,
    sigma,
    n_sims=50000,
    seed=42
):
    """
    European Call — Monte Carlo (risk-neutral GBM)
    """

    if seed is not None:
        np.random.seed(seed)

    # -------------------------
    # Draw random shocks
    # -------------------------
    Z = np.random.normal(size=n_sims)

    # -------------------------
    # Simulate terminal prices
    # -------------------------
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    S_T = S * np.exp(drift + diffusion)

    # -------------------------
    # Compute payoffs
    # -------------------------
    payoffs = np.maximum(S_T - K, 0)

    # -------------------------
    # Discount expected payoff
    # -------------------------
    price = np.exp(-r * T) * np.mean(payoffs)

    return price
