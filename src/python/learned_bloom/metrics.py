def false_positive_rate(false_positives: int, negatives: int) -> float:
    if negatives == 0:
        return 0.0
    return false_positives / negatives


def recall(true_positives: int, false_negatives: int) -> float:
    denom = true_positives + false_negatives
    if denom == 0:
        return 0.0
    return true_positives / denom
