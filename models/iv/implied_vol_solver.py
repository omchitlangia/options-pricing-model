from models.black_scholes import black_scholes_call

def implied_vol_bisection(
    market_price,
    S,
    K,
    T,
    r,
    tol=1e-6,
    max_iter=100
):
    """
    Computes implied volatility using the bisection method.
    """

    # -------------------------------
    # Volatility bounds
    # -------------------------------
    sigma_low = 1e-6
    sigma_high = 3.0

    # -------------------------------
    # Check intrinsic value
    # -------------------------------
    intrinsic = max(S - K * pow(2.718281828, -r * T), 0)

    if market_price < intrinsic:
        return None  # no valid implied volatility

    # -------------------------------
    # Bisection loop
    # -------------------------------
    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_low + sigma_high)

        price_mid = black_scholes_call(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma_mid
        )

        if abs(price_mid - market_price) < tol:
            return sigma_mid

        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    # Return best estimate after max iterations
    return 0.5 * (sigma_low + sigma_high)
