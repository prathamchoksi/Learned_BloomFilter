from dataclasses import dataclass


@dataclass
class LogisticModelConfig:
    C: float = 1.0
    max_iter: int = 1000
    threshold: float = 0.30


def predict_membership(score: float, threshold: float) -> bool:
    return score >= threshold
