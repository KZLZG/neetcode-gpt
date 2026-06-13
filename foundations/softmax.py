import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = z - max(z)
        sum_of_exps = np.exp(z).sum()
        answer = np.exp(z) / sum_of_exps
        return np.round(answer, 4)
