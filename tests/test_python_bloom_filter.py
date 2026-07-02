from learned_bloom.bloom_filter import BloomFilter


def test_inserted_key_is_found() -> None:
    bf = BloomFilter(expected_items=1000, target_fpr=0.01)
    bf.add("https://iitgn.ac.in")

    assert bf.contains("https://iitgn.ac.in") is True


def test_no_false_negatives_for_inserted_keys() -> None:
    bf = BloomFilter(expected_items=2000, target_fpr=0.01)
    keys = [f"https://example.org/item/{i}" for i in range(500)]
    bf.bulk_add(keys)

    missing = [k for k in keys if not bf.contains(k)]
    assert missing == []


def test_size_and_hash_parameters_are_positive() -> None:
    bf = BloomFilter(expected_items=5000, target_fpr=0.005)

    assert bf.num_bits > 0
    assert bf.num_hashes > 0
    assert bf.memory_bytes > 0


def test_estimated_fpr_in_valid_range() -> None:
    bf = BloomFilter(expected_items=1000, target_fpr=0.01)
    bf.bulk_add([f"x-{i}" for i in range(800)])

    est = bf.estimated_fpr()
    assert 0.0 <= est <= 1.0
