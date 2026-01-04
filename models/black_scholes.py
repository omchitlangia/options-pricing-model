import math

def normal_pdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def normal_cdf(x, steps=10_000):
    lower_bound = -10.0
    upper_bound = x
    step_size = (upper_bound - lower_bound) / steps

    area = 0.0
    current = lower_bound

    for _ in range(steps):
        next_point = current + step_size
        area += 0.5 * (normal_pdf(current) + normal_pdf(next_point)) * step_size
        current = next_point

    return area

def d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

def black_scholes_call(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    return (
        S * normal_cdf(D1)
        - K * math.exp(-r * T) * normal_cdf(D2)
    )

S = 100      # spot
K = 100      # strike
T = 1.0      # 1 year
r = 0.05     # 5%
sigma = 0.2  # 20%

price = black_scholes_call(S, K, T, r, sigma)
print("\nExample:\n" + f"S = {S}, K = {K}, T = {T}, r = {r}, σ = {sigma} \nPrice = {price}\n\n")

# ---- sanity tests ----

print("T → 0:", black_scholes_call(100, 100, 1e-6, 0.05, 0.2))
print("σ → 0:", black_scholes_call(100, 100, 1.0, 0.05, 1e-6))
print("Deep ITM:", black_scholes_call(200, 100, 1.0, 0.05, 0.2))
print("Deep OTM:", black_scholes_call(50, 100, 1.0, 0.05, 0.2))
print("Low vol:", black_scholes_call(100, 100, 1.0, 0.05, 0.1))
print("High vol:", black_scholes_call(100, 100, 1.0, 0.05, 0.3))
