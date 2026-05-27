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
        # return (np.round(w, 5), round(b, 5))
        length = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0
        while epochs:
            y_hat = X @ w + b
            MSE = np.sqrt((y_hat - y)**2)/length
            dw = (2.0/length) * (X.T @ (y_hat-y))
            db = (2.0/length)*np.sum(y_hat-y)

            w = w - lr*dw
            b = b - lr*db
            epochs-=1
        
        return (np.round(w, 5), round(b, 5))
