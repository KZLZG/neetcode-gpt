import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        def sigmoid(z):
            z = z - 1e-8
            return 1 / (1 + np.exp(-z))

        z = np.dot(x, w) + b
        y_hat = sigmoid(z)

        L = 0.5 * (y_hat - y_true)**2
        print(z, y_hat, L)
        dw = (y_hat - y_true) * y_hat * (1 - y_hat) * x
        db = (y_hat - y_true) * y_hat * (1 - y_hat)

        return np.round(dw, 5), round(db, 5)


