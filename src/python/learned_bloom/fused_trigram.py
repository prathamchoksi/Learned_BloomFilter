from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix

UINT64_MASK = (1 << 64) - 1
ROLLING_BASE = 1315423911
DEFAULT_N_FEATURES = 2**18


@dataclass(frozen=True)
class FusedTrigramModelMetadata:
    n_features: int
    analyzer: str
    ngram_range: tuple[int, int]
    hash_scheme: str
    threshold: float
    feature_type: str = "char_trigram"


def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def rolling_update(prev_hash: int, left_char: str, right_char: str) -> int:
    left_value = ord(left_char) & UINT64_MASK
    right_value = ord(right_char) & UINT64_MASK
    base_sq = (ROLLING_BASE * ROLLING_BASE) & UINT64_MASK
    updated = (prev_hash - ((left_value * base_sq) & UINT64_MASK)) & UINT64_MASK
    updated = (updated * ROLLING_BASE + right_value) & UINT64_MASK
    return updated


def trigram_hash(text: str, start: int = 0) -> int:
    token = text[start : start + 3]
    h = 0
    for ch in token:
        h = (h * ROLLING_BASE + (ord(ch) & UINT64_MASK)) & UINT64_MASK
    return h


def trigram_indices(text: str, n_features: int) -> np.ndarray:
    normalized = normalize_text(text)
    if len(normalized) < 3:
        return np.empty(0, dtype=np.int64)

    indices = np.empty(len(normalized) - 2, dtype=np.int64)
    current_hash = trigram_hash(normalized, 0)
    indices[0] = current_hash % n_features

    for idx in range(3, len(normalized)):
        current_hash = rolling_update(current_hash, normalized[idx - 3], normalized[idx])
        indices[idx - 2] = current_hash % n_features

    return indices


def build_trigram_matrix(urls: Iterable[str], n_features: int) -> csr_matrix:
    indices_buffer: list[int] = []
    data_buffer: list[float] = []
    indptr = [0]

    for url in urls:
        trigram_indices_array = trigram_indices(url, n_features)
        if trigram_indices_array.size == 0:
            indptr.append(len(indices_buffer))
            continue

        unique_indices, counts = np.unique(trigram_indices_array, return_counts=True)
        indices_buffer.extend(unique_indices.astype(np.int64).tolist())
        data_buffer.extend(counts.astype(np.float64).tolist())
        indptr.append(len(indices_buffer))

    return csr_matrix(
        (
            np.asarray(data_buffer, dtype=np.float64),
            np.asarray(indices_buffer, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(indptr) - 1, n_features),
        dtype=np.float64,
    )


def export_fused_model(
    output_dir: Path,
    coef: np.ndarray,
    intercept: float,
    metadata: FusedTrigramModelMetadata,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.bin"
    metadata_path = output_dir / "metadata.json"

    coef_1d = np.asarray(coef, dtype=np.float64).reshape(-1)
    if coef_1d.size != metadata.n_features:
        raise ValueError(
            f"Expected {metadata.n_features} coefficients, got {coef_1d.size}"
        )

    with weights_path.open("wb") as f:
        np.asarray([metadata.n_features], dtype=np.uint64).tofile(f)
        np.asarray([intercept], dtype=np.float64).tofile(f)
        coef_1d.tofile(f)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_features": metadata.n_features,
                "analyzer": metadata.analyzer,
                "ngram_range": [metadata.ngram_range[0], metadata.ngram_range[1]],
                "hash_scheme": metadata.hash_scheme,
                "threshold": metadata.threshold,
                "feature_type": metadata.feature_type,
            },
            f,
            indent=2,
        )

    return weights_path, metadata_path
