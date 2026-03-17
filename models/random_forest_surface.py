"""
random_forest_surface.py
-------------------------
Random Forest model for implied volatility surface fitting.

Expected interface:
    model = RandomForestSurfaceModel(n_estimators=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

Input features (X):
    moneyness, log_moneyness, time_to_maturity, sqrt_T,
    moneyness_T_interaction, ticker_encoded

Target (y):
    implied_vol  (float, annualized)

TODO (Step 8.3): implement fit / predict / evaluate / feature_importance methods
"""


class RandomForestSurfaceModel:
    """Random Forest ensemble model for IV surface fitting."""

    def __init__(self, n_estimators: int = 200, max_depth: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None

    def fit(self, X, y):
        raise NotImplementedError("RandomForestSurfaceModel.fit — coming in Step 8.3")

    def predict(self, X):
        raise NotImplementedError("RandomForestSurfaceModel.predict — coming in Step 8.3")

    def feature_importance(self):
        raise NotImplementedError("RandomForestSurfaceModel.feature_importance — coming in Step 8.3")
