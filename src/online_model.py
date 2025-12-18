from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Dict

import numpy as np


@dataclass
class OnlineMahalanobisADConfig:
    window_size: int = 100           # number of recent points to model "normal"
    min_window: int = 30             # minimum before decisions are meaningful
    contamination: float = 0.03      # expected anomaly proportion
    eps: float = 1e-6                # numerical stability


class OnlineMahalanobisAD:
    """
    Lightweight online anomaly detector for multivariate streams.

    - Maintains a sliding window of recent observations.
    - Estimates mean and covariance on the window.
    - Uses Mahalanobis distance as an anomaly score.
    - Sets an adaptive threshold based on a high percentile
      of past distances (approximate contamination control).
 """

    def __init__(self, n_features: int, config: Optional[OnlineMahalanobisADConfig] = None):
        self.config = config or OnlineMahalanobisADConfig()
        self.n_features = n_features
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self.dist_history: Deque[float] = deque(maxlen=self.config.window_size)

    def _fit_window(self) -> Dict[str, np.ndarray]:
        """Estimate mean and covariance on the current buffer."""
        X = np.stack(self.buffer, axis=0)  # (n, d)
        mu = X.mean(axis=0)
        # unbiased covariance, rowvar=False since rows are observations
        cov = np.cov(X, rowvar=False)
        # regularize covariance for numerical stability
        cov = cov + np.eye(self.n_features) * self.config.eps
        cov_inv = np.linalg.pinv(cov)
        return {"mu": mu, "cov_inv": cov_inv}

    def _mahalanobis_sq(self, x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
        """Squared Mahalanobis distance of x from (mu, cov)."""
        diff = x - mu
        return float(diff.T @ cov_inv @ diff)

    def update(self, x: np.ndarray) -> Dict[str, float]:
        """
        Ingest a new observation x and return:
        - score: squared Mahalanobis distance
        - threshold: current adaptive threshold
        - is_anomaly: 1 if anomalous, 0 otherwise
        """
        assert x.shape[0] == self.n_features, "Input dimension mismatch"

        # add to buffer
        self.buffer.append(x)

        # not enough data yet
        if len(self.buffer) < self.config.min_window:
            self.dist_history.append(0.0)
            return {"score": 0.0, "threshold": float("inf"), "is_anomaly": 0}

        params = self._fit_window()
        score = self._mahalanobis_sq(x, params["mu"], params["cov_inv"])

        # update history & threshold
        self.dist_history.append(score)
        if len(self.dist_history) < self.config.min_window:
            threshold = float("inf")
        else:
            # high percentile as adaptive boundary for normal behaviour
            alpha = 100 * (1.0 - self.config.contamination)
            threshold = float(np.percentile(list(self.dist_history), alpha))

        is_anomaly = 1 if score > threshold else 0
        return {"score": score, "threshold": threshold, "is_anomaly": is_anomaly}
