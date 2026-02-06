import math

def binomial_call_price(S, K, T, r, sigma, steps=100):
    """
    European Call Option — CRR Binomial Tree
    """

    dt = T / steps

    # up/down factors
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u

    # risk-neutral probability
    p = (math.exp(r * dt) - d) / (u - d)

    # discount factor per step
    disc = math.exp(-r * dt)

    # -------------------------------
    # Terminal stock prices
    # -------------------------------
    prices = []
    for j in range(steps + 1):
        price = S * (u ** (steps - j)) * (d ** j)
        prices.append(price)

    # -------------------------------
    # Terminal payoffs
    # -------------------------------
    values = [max(price - K, 0) for price in prices]

    # -------------------------------
    # Backward induction
    # -------------------------------
    for _ in range(steps):
        values = [
            disc * (p * values[i] + (1 - p) * values[i + 1])
            for i in range(len(values) - 1)
        ]

    return values[0]
