Part 1 — Why introduce Monte Carlo pricing
A third pricing engine is introduced to add an independent validation layer beyond closed-form (Black–Scholes) and lattice (binomial tree) methods. Even if those two agree, they are still structurally related, so a simulation-based approach gives a stronger cross-check. Monte Carlo helps eliminate the doubt that pricing agreement is coming from similar model structure or discretization logic. If simulation-based pricing — built from random path generation — produces the same prices under the same implied volatility input, it confirms that results are not tied to a specific computational framework. Monte Carlo is fundamentally different because it prices options by simulating many possible future price paths and averaging discounted payoffs, instead of using a formula (Black–Scholes) or a recombining tree (binomial). It is path-based rather than formula-based or lattice-based.

Part 2 — Core idea of Monte Carlo valuation
Monte Carlo pricing is trying to estimate the expected payoff of the option under risk-neutral dynamics, discounted back to today. That expected discounted payoff is the option’s fair price. Instead of computing the expectation using a closed formula or analytic integration, Monte Carlo computes it numerically by simulating many possible future price paths, calculating the payoff for each path, and then taking the average. Random sampling replaces integration in Monte Carlo. Rather than integrating over all possible outcomes mathematically, we approximate the expectation by averaging results across a large number of simulated scenarios.

Part 3 — Risk-Neutral GBM Terminal Distribution
The underlying price is assumed to follow Geometric Brownian Motion (GBM) under the risk-neutral measure. Returns are lognormal and the expected growth rate equals the risk-free rate (not historical drift). 
Formula used -> S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
Z is a standard normal random variable (N(0,1)). Each simulation draws a different Z, which creates different terminal prices across paths.

Part 4 — Why the Drift Term is (r − ½σ²)
The −½σ² term appears due to Ito’s correction when converting the stochastic process from log returns to price levels. Because the exponential of a random variable introduces upward bias, this term adjusts the drift so that the simulated price process remains correctly scaled.
Under the risk-neutral measure, the expected future stock price must grow at the risk-free rate:
E[S_T] = S_0 * exp(rT)
The −½σ² term ensures that the simulated GBM process satisfies this expectation condition.
If the −½σ² correction is removed, the simulated expected price becomes too large. This breaks the risk-neutral condition and causes Monte Carlo pricing to systematically overestimate option values.

Part 5 — Simulation Workflow
1. Draw a random standard normal variable \( Z \sim N(0,1) \)  
2. Compute the terminal stock price using the GBM formula  
   S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
3. Compute the option payoff at maturity  
   payoff = max(S_T - K, 0)   (for a call option)
4. Store this payoff
This process is repeated for many simulations.
In this implementation, only the terminal price is simulated directly using the GBM terminal distribution. Full paths are not required for European options because the payoff depends only on the final price.
Each simulation produces one payoff. After running many simulations:
1. Compute the average payoff across all simulations  
2. Discount the average payoff back to today using the risk-free rate  
Option Price = exp(-rT) * Average(payoff)

Part 6 — Law of Large Numbers & Convergence
Monte Carlo pricing relies on the Law of Large Numbers. As the number of simulations increases, the average of the simulated payoffs converges to the true expected payoff under the risk-neutral measure.
Monte Carlo error decreases at a rate proportional to:
error ≈ 1 / √N
where N is the number of simulations. Doubling accuracy therefore requires roughly four times as many simulations.
Monte Carlo relies on random sampling, so each run introduces sampling variance. Binomial pricing is deterministic and structured, so it converges smoothly as the number of steps increases. Monte Carlo estimates fluctuate around the true value until enough simulations are used.

Part 7 — Random Seed and Reproducibility 
Fixing the random seed initializes the random number generator in a deterministic way. This means the same sequence of random numbers will be produced every time the code runs. It allows results to be exactly reproducible. During development and debugging, this makes it easier to verify changes in the code because differences in output come from the code itself, not from different random draws.The seed is usually removed or randomized when running final large simulations or production experiments, so that results reflect natural randomness instead of a fixed sequence of draws.

Part 8 — Path Simulation vs Terminal Sampling 
Terminal-only simulation generates the final price \(S_T\) directly using the GBM terminal formula. Full path simulation generates the entire price trajectory over time by simulating many small steps between \(S_0\) and \(S_T\). European options only require the terminal price because the payoff depends only on the final value at maturity. Therefore, terminal sampling is sufficient for pricing. Path simulation helps visualize how stochastic price diffusion behaves over time. Even though pricing only needs \(S_T\), plotting full paths provides intuition about volatility, randomness, and dispersion in the simulated process.

Part 9 — Cross-Model Consistency Conclusion
Monte Carlo prices were nearly identical to binomial and Black–Scholes prices when implied volatility was used as the model input. Pricing errors collapsed near zero across both maturity and moneyness buckets. The choice of pricing engine (closed-form, lattice, or simulation) had little effect on calibrated option prices. Once implied volatility was used, all three approaches produced consistent results.The experiments confirm that pricing accuracy is driven primarily by the volatility specification rather than the numerical pricing method. Structured mispricing observed earlier under constant volatility assumptions disappears when volatility is aligned with market-implied levels.
Consistent prices across analytic, lattice, and simulation methods → volatility input dominates pricing accuracy.
