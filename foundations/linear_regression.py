import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        pred = np.dot(X, weights)

        # Round to 5 decimal places
        return np.round(pred, 5)
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        n = len(model_prediction)
        squared_errors = (model_prediction - ground_truth)**2
        mse = 1/n * np.sum(squared_errors)

        return round(float(mse), 5)
