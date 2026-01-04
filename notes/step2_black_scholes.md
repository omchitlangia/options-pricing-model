Part 1 — What problem does Black–Scholes solve?
Options are harder to price than stocks because of the asymmetric payoffs limiting the potential downside risk. We can't really depend on the expected returns as options rewards uncertainity and not correct direction and also market predictions don't matter as we use risk-neutral probablities to price the option. We also care about no-arbitrage pricing, it states that if the payoffs of the two portfolios are same, it should have the same price, using this principle we price the option using Black Scholes equation on bonds and the underlying asset to replicate the option payoffs.

Part 2 — Risk-Neutral Pricing vs Real-World Pricing
The risk-neutral measure (Q) is a made-up probability that assumes that every asset yields the risk-free rate, and risk-neutral pricing creates a linear combination of bonds and the underlying asset to replicate the payoff produced by an option and then price it in a no-arbitrage environment and then discounted at the risk-free rate. The drift disappears as it is hedged away by the replicated portfolio of bonds and the asset itself and what is left is volatility. Probablities in Black Scholes is not beliefs in the market but the functions of volatility-adjusted and risk-volatility adjusted distance between the asset price and the strike price.

Part 3 — Distribution of Future Prices
About the price dynamics, it is assumed that returns scale with price, the spot price can never go negative and trading is continous, there are no gap ups or gap downs, to keep the hedging also continuous. We take log-prices instead of normal prices to convert the multiplicative growth to additive growth. The normal distribution appears in the formula because in the Geometric Brownian Motion the weight is normal, hence the log-prices are normal, making the prices log-normal.

Part 4 — Meaning of d1 and d2
d2 intuitively measures the volatility-adjusted distance between the spot and the strike price. d1 is different from d2 as d1 is the same distance d2 adjusted for its sensitivity of option price with the spot price. In the Black-Scholes equation N(d2) shows the risk-neutral probability of option expiring in the money and N(d1) shows how the option price reacts with unit change in the spot price and is used for hedging.

Part 5 — Economic Interpretation of the Call Price Formula
S multiplied by N(d1) represents the present value of receiving the stock, weighted by how much the option’s value responds to changes in the spot price. strike discounted and multiplied by N(d2) represents the present value of the expected strike payment, weighted by the risk-neutral probability that the option expires in the money. A call option gives the right to receive the stock while paying the strike, so its value is the value of receiving the stock minus the expected cost of paying the strike.

Part 6 — Put–Call Parity
Put-Call parity is a no-arbitrage relationship that shows that a long call and a short put replicates the payoff of longing the asset and borrowing cash equal to strike discounted to present. Volatility does not affect the call put parity because the effect cancels off due to same parameters in the call and put. This only holds for european options as you can only exercise the option when it expires meaning the discounting of the strike makes sense, whereas in American options, people can exercise their option before expiry, early exercise breaks the fixed terminal payoff structure.

Part 7 — Role of Volatility
Higher volatility increases the option value because higher uncertainity means increased expected payoff due to convexity. “Options reward uncertainty, not direction”, means that uncertainity in any direction will be better than chopping around the strike price, as the drawdown is limited to the premium paid and profits are theoritically unlimited. Convexity comes in the picture for the same reason, as on an option, no one can lose more than the premium paid for the option, but they can make much larger profits, making the payoffs asymmetrical in nature.

Part 8 — Limiting & Sanity Cases
    T→0, implies that the option is almost expired making the option price = max(spot - strike, 0)
    σ→0, implies that there is no volatility => no randomness, hence making the option price = spot - discounted strike.
    Deep ITM Call -> The options sells at a premium, spot - strike discounted
    Deep OTM Call -> The price goes to 0.
    Increasing volatility -> Higher price than usual, as the uncertainity is higher and higher expected payoff due to convexity.

Part 9 — What Black–Scholes Gets Right and What It Misses
BS is still useful despite unrealistic assumptions, because it helps price the options without taking into belief any individual opinion or realistic market probabilities. It sets a benchmark by using the no-arbitrage model and a consistent pricing system using replication. The assumption that I feel is the most problematic is transaction costs, that creates infinite rebalancing and continuous trading hours. I expect BS to misprice American options, options during stressed markets, deep OTM options and short/long dated options.

Part 10 — Key Takeaways from Step 2
I understood what is options and what are the key parameters used to price it. I learned about different types of options and their implemnetations too. The intuition that surprised me the most would be creating the same payoffs using replication and also the logic used behind the put call parity. This is a strong baseline model and it creates a solid no-arbitrage foundation for the upcoming computational models.