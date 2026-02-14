To complement the numerical analysis, visual diagnostics were used to examine the structure of Black–Scholes pricing errors and the behavior of implied volatility across strikes and maturities.

The plot of Black–Scholes pricing error versus time to maturity shows a systematic increase in underpricing as maturity increases, indicating that the constant volatility assumption becomes increasingly restrictive over longer horizons. Errors also exhibit greater dispersion for longer-dated options, reflecting accumulating uncertainty.

The pricing error versus moneyness plot reveals a smooth, non-random structure, with minimal error for deep out-of-the-money options and significant underpricing around at-the-money and in-the-money regions. This pattern suggests that volatility cannot be treated as constant across strikes and directly motivates a strike-dependent volatility representation.

Implied volatility plotted against moneyness exhibits a clear smile-shaped pattern, with minimum implied volatility near at-the-money options and increasing implied volatility away from the center. This demonstrates that market prices embed asymmetric risk perceptions and tail-event sensitivity.

Finally, implied volatility versus maturity shows a downward-sloping term structure, with higher implied volatility for shorter maturities. This indicates that near-term uncertainty is priced more aggressively by the market, further violating the constant-volatility assumption of Black–Scholes.

Together, these visual diagnostics reinforce the numerical findings and confirm that systematic pricing errors arise from volatility being a function of strike and maturity rather than a single constant parameter.

Binomial pricing errors are tightly centered around zero across maturities, with no systematic maturity-dependent bias, confirming numerical convergence and correct implied volatility calibration.

Binomial pricing error dispersion is slightly higher near at-the-money options, consistent with higher Gamma and curvature sensitivity, but remains economically negligible, indicating stable binomial convergence across strikes.