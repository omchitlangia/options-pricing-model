"""
mlp_surface.py
---------------
Multi-layer perceptron (neural network) for IV surface fitting.

Expected interface:
    model = MLPSurfaceModel(hidden_layers=[64, 64])
    model.fit(X_train, y_train, epochs=100)
    y_pred = model.predict(X_test)

Input features (X):
    moneyness, log_moneyness, time_to_maturity, sqrt_T,
    moneyness_T_interaction, ticker_encoded

Target (y):
    implied_vol  (float, annualized)

TODO (Step 8.3): implement with PyTorch or scikit-learn MLPRegressor
"""


class MLPSurfaceModel:
    """Multi-layer perceptron model for IV surface fitting."""

    def __init__(self, hidden_layers: list = None):
        self.hidden_layers = hidden_layers or [64, 64]
        self._model = None

    def fit(self, X, y, epochs: int = 100):
        raise NotImplementedError("MLPSurfaceModel.fit — coming in Step 8.3")

    def predict(self, X):
        raise NotImplementedError("MLPSurfaceModel.predict — coming in Step 8.3")
