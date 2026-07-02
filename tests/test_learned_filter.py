from learned_bloom.learned_filter import LearnedBloomFilter


def test_model_positive_short_circuit() -> None:
    lbf = LearnedBloomFilter(model_threshold=0.3)
    assert lbf.query(score=0.9, backup_contains=False) is True


def test_backup_used_when_model_negative() -> None:
    lbf = LearnedBloomFilter(model_threshold=0.3)
    assert lbf.query(score=0.1, backup_contains=True) is True
    assert lbf.query(score=0.1, backup_contains=False) is False
