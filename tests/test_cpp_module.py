from __future__ import annotations

import struct
from pathlib import Path

import pytest


cpp = pytest.importorskip("learned_bloom_cpp")


def _write_minimal_weights(path: Path, n_features: int = 8, intercept: float = 0.0) -> None:
    weights = [0.0] * n_features
    with path.open("wb") as f:
        f.write(struct.pack("<Qd", n_features, intercept))
        f.write(struct.pack("<" + "d" * n_features, *weights))


def test_cpp_bloom_filter_roundtrip() -> None:
    bloom_filter = cpp.BloomFilter(128, 3)
    bloom_filter.add("example.com")

    assert bloom_filter.contains("example.com") is True


def test_cpp_fused_trigram_scorer_loads_and_scores(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.bin"
    _write_minimal_weights(weights_path)

    scorer = cpp.FusedTrigramScorer(str(weights_path))

    assert scorer.score("http://example.com") == 0.0
    assert scorer.predict("http://example.com", 0.5) is False