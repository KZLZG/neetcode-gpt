import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        b = 0
        w = np.zeros(shape=X.shape[1])
        for epoch in range(epochs):
            
            preds = np.empty(shape=y.shape)
            preds = X @ w + b

            errors = preds - y
            mse = np.sum(errors**2) / len(y)
            print(mse)

            dL = 2/len(y) * X.T @ errors  
            db = 2/len(y) * np.sum(errors)
            w = w - lr*dL
            b = b - lr*db
            print(b)
        return (w.round(5), b.round(5))
