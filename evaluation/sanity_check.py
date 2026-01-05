from models.black_scholes import black_scholes_call

print("Sanity Tests")
print("T → 0:", black_scholes_call(100, 100, 1e-6, 0.05, 0.2))
print("σ → 0:", black_scholes_call(100, 100, 1.0, 0.05, 1e-6))
print("Deep ITM:", black_scholes_call(200, 100, 1.0, 0.05, 0.2))
print("Deep OTM:", black_scholes_call(50, 100, 1.0, 0.05, 0.2))
print("Low vol:", black_scholes_call(100, 100, 1.0, 0.05, 0.1))
print("High vol:", black_scholes_call(100, 100, 1.0, 0.05, 0.3))
