Section 1 — What problem are we solving?
In this project, we are going to price options fairly using the classical methods like Black Scholes, Binomial Tree and Monte Carlo, in the US liquid options. We are going to look at the assumptions that we consider in our project and we will see how it fails or explains the price. We will also be creating practical results instead of just valuing the assets, and optimise the project for for risk management and hedging purposes.
Option pricing is typically hard because the payoffs are asymmetric in nature unlike futures trading. We have a choice and not an obligation. Plus, the major downside risk is just the premium paid but the maximum gains can be theoretically unlimited.

Section 2 — Why options can be priced without predicting markets?
Options can be priced without predicting the markets by replicating the returns using the spot price and the risk-free rate of returns or the T-bill returns, we create a combination and find risk-neutral probabilities to match the options return without predicting the market. We use the risk-neutral pricing method to price the asset and not predict it. Here, we also use set payoffs, as they are linear in replication, removing drift from the equation.

Section 3 — Why volatility is the key unknown?
Volatility is the key unknown because it is dependent on many factors like news, wars etc. and is not constant. Under risk-neutral pricing, uncertainty is summarized by volatility or the standard deviation because options reward uncertainty and not direction. The move should be big enough or distant from the mean, even if the mean is exactly the same. More or less, every other variable is known (like risk-free rate, strike price, expiry date, and the spot price), but volatility/variance is unknown and everything comes down to that.

Section 4 — When Black–Scholes will fail?
i. If the volatility is not constant
ii. If the trading hours are not continous and there are gap up or gap down openings.
iii. Market frictions such as latency, liquidity constraints, and discrete trading prevent continuous rebalancing of the replicating portfolio, violating the no-arbitrage framework assumed by the model.
iv. If transaction costs exist, there will be infinite rebalancings.
v. If asset prices exhibit jumps or fat-tailed return distributions, the log-normal price assumption of Black–Scholes fails to capture extreme market movements.
