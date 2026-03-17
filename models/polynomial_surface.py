"""
polynomial_surface.py
----------------------
Polynomial regression baseline for implied volatility surface fitting.

Expected interface:
    model = PolynomialSurfaceModel(degree=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

Input features (X):
    moneyness, log_moneyness, time_to_maturity, sqrt_T, moneyness_T_interaction

Target (y):
    implied_vol  (float, annualized)

TODO (Step 8.3): implement fit / predict / evaluate methods
"""


class PolynomialSurfaceModel:
    """Polynomial regression model for IV surface fitting."""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self._model = None

    def fit(self, X, y):
        raise NotImplementedError("PolynomialSurfaceModel.fit — coming in Step 8.3")

    def predict(self, X):
        raise NotImplementedError("PolynomialSurfaceModel.predict — coming in Step 8.3")
