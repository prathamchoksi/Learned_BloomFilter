from learned_bloom.metrics import false_positive_rate, recall


def test_metrics_fpr_zero_negatives() -> None:
    assert false_positive_rate(0, 0) == 0.0


def test_metrics_recall_basic() -> None:
    assert recall(true_positives=9, false_negatives=1) == 0.9
