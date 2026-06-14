import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mu = x.mean()
        print(mu)
        sigma = x.var() # ((x - mu)**2).mean()
        print(sigma)
        answer = ((x - mu) * gamma) / np.sqrt(sigma + 1e-5) + beta
        print(answer)
        return np.round(answer, 5)
