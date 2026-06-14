import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool,
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform:
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        y = None
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            y = gamma * x_hat + beta

            running_mean = ((1 - momentum) * np.array(running_mean)) + (batch_mean * momentum)
            running_var = ((1 - momentum) * np.array(running_var) + (batch_var * momentum))
        else:
            x_hat = (x - np.array(running_mean)) / np.sqrt(np.array(running_var) + eps)
            y = x_hat
        return (y.round(4), np.round(running_mean, 4).tolist(), np.round(running_var, 4))
