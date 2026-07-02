from __future__ import annotations

from pathlib import Path
import json

import numpy as np

UINT64_MASK = (1 << 64) - 1
ROLLING_BASE = 1315423911


class FusedTrigramScorerPy:
    """Pure Python fallback for fused trigram scoring without C++ compilation."""

    def __init__(self, weights_path: str) -> None:
        self.weights_path = Path(weights_path)
        self.n_features = 0
        self.intercept = 0.0
        self.weights: np.ndarray = np.array([], dtype=np.float64)

        self._load_weights()

    def _load_weights(self) -> None:
        with self.weights_path.open("rb") as f:
            n_feat_arr = np.fromfile(f, dtype=np.uint64, count=1)
            intercept_arr = np.fromfile(f, dtype=np.float64, count=1)
            weights_arr = np.fromfile(f, dtype=np.float64)

            if len(n_feat_arr) < 1:
                raise RuntimeError("Failed to read n_features from weights file")
            if len(intercept_arr) < 1:
                raise RuntimeError("Failed to read intercept from weights file")

            self.n_features = int(n_feat_arr[0])
            self.intercept = float(intercept_arr[0])
            self.weights = weights_arr

            if self.weights.size != self.n_features:
                raise RuntimeError(
                    f"Weight array size {self.weights.size} does not match n_features {self.n_features}"
                )

    @staticmethod
    def normalize_url(url: str) -> str:
        return str(url).strip().lower()

    @staticmethod
    def trigram_hash(text: str, start: int = 0) -> int:
        token = text[start : start + 3]
        h = 0
        for ch in token:
            h = (h * ROLLING_BASE + ord(ch)) & UINT64_MASK
        return h

    @staticmethod
    def rolling_update(prev_hash: int, left_char: str, right_char: str) -> int:
        base_sq = (ROLLING_BASE * ROLLING_BASE) & UINT64_MASK
        removed = (ord(left_char) * base_sq) & UINT64_MASK
        updated = ((prev_hash - removed) & UINT64_MASK) * ROLLING_BASE + ord(right_char)
        return updated & UINT64_MASK

    def score(self, url: str) -> float:
        normalized = self.normalize_url(url)
        if len(normalized) < 3 or self.n_features == 0:
            return self.intercept

        logit = self.intercept
        trigram = self.trigram_hash(normalized, 0)
        logit += self.weights[trigram % self.n_features]

        for i in range(3, len(normalized)):
            trigram = self.rolling_update(trigram, normalized[i - 3], normalized[i])
            logit += self.weights[trigram % self.n_features]

        return logit

    def predict(self, url: str, threshold: float = 0.5) -> bool:
        logit = self.score(url)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob >= threshold
