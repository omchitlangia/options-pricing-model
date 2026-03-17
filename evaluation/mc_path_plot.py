"""
mc_path_plot.py
----------------
Visualises Monte Carlo simulated GBM paths.

Run from the project root:
    python -m evaluation.mc_path_plot
"""

import numpy as np
import matplotlib.pyplot as plt

S0 = 100
T = 1.0
r = 0.05
sigma = 0.2

steps = 252
paths = 50
dt = T / steps

np.random.seed(42)

time = np.linspace(0, T, steps + 1)
all_paths = np.zeros((paths, steps + 1))
all_paths[:, 0] = S0

for i in range(paths):
    Z = np.random.normal(size=steps)
    for t in range(steps):
        all_paths[i, t + 1] = all_paths[i, t] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t]
        )

plt.figure(figsize=(8, 5))
for i in range(paths):
    plt.plot(time, all_paths[i], alpha=0.6)
plt.title("Monte Carlo Simulated GBM Paths")
plt.xlabel("Time (Years)")
plt.ylabel("Price")
plt.show()
