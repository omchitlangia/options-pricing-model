Part 1 — Objective of Step 3
It is not enough to implement Black-Scholes mathematically, because of the unrealistic assumptions that we take into account when calculating the option prices. Step 3 specifically tries to implement Black-Scholes model to real data and analyse the error. Step 3 is different from step 2, as, in step 2, we only formed a model for BS and run sanity tests on it, whereas, in step 3 we will analyse the market mispricing relative to the math on real life data.

Part 2 — Choice of Asset and Options
We are choosing SPY (S&P500 ETF), as the underlying asset for this step. This is because SPY options are highly liquid, has tight bid–ask spreads and deep option chains are available with different strikes and maturities. Also, SPY is a standard academic research equity, with well studied volatility behaviour. First, we are choosing only call options for testing and the put prices can be easily calculated using put-call parity. Expiries are restricted to short maturities because in the long dated options volatility does not remain constant because of regime change and other macroeconomic factors, whereas Black–Scholes assumes constant volatility over the option’s life. The longer the option the more unrealistic the assumption becomes.

Part 3 — Market Data Collection
The option data is fetched using the SPY option chain available on Yahoo Finance. "Raw market data" in our context means fetching the available option chain with all the strikes and expiries without filtering anything. It is important not to filter at this stage to avoid introducing selection bias and to ensure that any observed mispricing reflects genuine model failure rather than data preprocessing choices.

Part 4 — Data Cleaning and Filtering
Illiquid options are removed because their prices are often stale, noisy, or distorted by wide bid–ask spreads, which would introduce pricing errors unrelated to the Black–Scholes model itself. The mid-price is used as it best approximates the market’s consensus fair value, whereas bid and ask prices reflect execution frictions rather than theoretical option value. Moneyness is restricted to avoid deep ITM and OTM options, whose prices are dominated by intrinsic value, tail risk, or additional effects (dividends, rates, liquidity) that Black–Scholes does not model well. This step prevents market microstructure noise and data artifacts from being misinterpreted as failures of the Black–Scholes model.

Part 5 — Choice of Volatility Input
Black–Scholes requires volatility as an input because volatility tells the model how much the price of the asset can move in the future. The more the price can move around, the more valuable an option becomes. Historical volatility was used because it is easy to calculate from past prices and does not depend on option prices themselves. This allows us to test Black–Scholes without feeding it information taken from the same market we are trying to evaluate. We framed historical volatility on the basis of past 90 days data from Yahoo Finance. A constant volatility was intentionally used to keep the model simple and to clearly see where it fails. Since real markets do not have the same volatility for all options, this assumption creates visible pricing errors that can be studied.

Part 6 — Pricing Methodology
Black-Scholes pricing was calculated from each option by first finding d1 and d2 and then their Normal CDF functions using scipy.norm.cdf, then using the current spot price time N(d1) subtracted by the strike price times N(d2) discounted to present. The inputs taken directly from market are the risk-free rate, spot price, strike price and days-to-expiry. Pricing error was defined for each option as the difference between the Black–Scholes price and the option’s market mid-price. Relative error was defined as this difference divided by the market mid-price. These errors were then averaged across options for analysis.

Part 7 — Overall Model Performance
Absolute error represents the average dollar amount deviation from the Black-Scholes pricing in any direction. The error was economically meaningful as it shows how much are we mispricing an option on average, with respect to the market price and whether the error is large enough to matter in practice This metric alone does not explain model failure as we have yet not analysed the behaviour for different types of options, like OTM, ITM, short-dated, long-dated etc, and hence we can not draw a conclusion based on the absolute error.

Part 8 — Error Analysis by Moneyness
Moneyness is defined as the ratio of the spot price to the strike price. Pricing error for OTM options were near 0, for ATM the options were underpriced and for ITM the intensity of underpricing increased further. These results indicate that the constant volatility assumption in Black–Scholes does not hold across different levels of moneyness. In real markets, options closer to or inside the money tend to be priced with higher effective volatility than what a single historical volatility can capture. As a result, Black–Scholes underestimates option values where volatility risk matters most.

Part 9 — Error Analysis by Maturity
We chose options with date-to-expiry of 7 - 45 days. We further grouped these data as short, medium and long dated options. The dollar amount error kept increasing with increasing maturity. This pattern implies that assuming a single constant volatility becomes increasingly unrealistic over longer horizons, as volatility changes over time and longer-dated options are more sensitive to these changes.

Part 10 — Key Findings
Pricing errors vary systematically with moneyness and maturity, showing that Black–Scholes fails due to its constant-volatility assumption rather than random noise.

Part 11 — Motivation for Next Step
Since volatility clearly varies across strikes and time, the next step is to extract volatility from market prices instead of assuming it.