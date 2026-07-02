import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BloomFilterConfig:
    expected_items: int
    target_fpr: float


class BloomFilter:
    """Baseline Bloom filter with double hashing and packed byte storage."""

    def __init__(self, expected_items: int, target_fpr: float) -> None:
        if expected_items <= 0:
            raise ValueError("expected_items must be > 0")
        if not (0.0 < target_fpr < 1.0):
            raise ValueError("target_fpr must be in (0, 1)")

        self.expected_items = expected_items
        self.target_fpr = target_fpr
        self.num_bits = self.optimal_num_bits(expected_items, target_fpr)
        self.num_hashes = self.optimal_num_hashes(self.num_bits, expected_items)

        self._bits = bytearray((self.num_bits + 7) // 8)
        self._items_added = 0

    @staticmethod
    def optimal_num_bits(expected_items: int, target_fpr: float) -> int:
        # m = -(n * ln(p)) / (ln(2)^2)
        m = -(expected_items * math.log(target_fpr)) / (math.log(2) ** 2)
        return max(8, math.ceil(m))

    @staticmethod
    def optimal_num_hashes(num_bits: int, expected_items: int) -> int:
        # k = (m / n) * ln(2)
        k = (num_bits / expected_items) * math.log(2)
        return max(1, math.ceil(k))

    @property
    def memory_bytes(self) -> int:
        return len(self._bits)

    @property
    def items_added(self) -> int:
        return self._items_added

    def add(self, key: str) -> None:
        for idx in self._hash_indexes(key):
            self._set_bit(idx)
        self._items_added += 1

    def bulk_add(self, keys: list[str]) -> None:
        for key in keys:
            self.add(key)

    def contains(self, key: str) -> bool:
        for idx in self._hash_indexes(key):
            if not self._test_bit(idx):
                return False
        return True

    def __contains__(self, key: str) -> bool:
        return self.contains(key)

    def estimated_fpr(self) -> float:
        # p = (1 - e^(-kn/m))^k
        exp_term = math.exp(-self.num_hashes * self._items_added / self.num_bits)
        return (1.0 - exp_term) ** self.num_hashes

    def _hash_indexes(self, key: str) -> list[int]:
        data = key.encode("utf-8", errors="ignore")

        h1 = int.from_bytes(hashlib.sha256(data).digest()[:8], "little", signed=False)
        h2 = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=False)

        if h2 == 0:
            h2 = 0x9E3779B185EBCA87

        return [int((h1 + i * h2) % self.num_bits) for i in range(self.num_hashes)]

    def _set_bit(self, bit_index: int) -> None:
        byte_index = bit_index // 8
        offset = bit_index % 8
        self._bits[byte_index] |= 1 << offset

    def _test_bit(self, bit_index: int) -> bool:
        byte_index = bit_index // 8
        offset = bit_index % 8
        return (self._bits[byte_index] & (1 << offset)) != 0
