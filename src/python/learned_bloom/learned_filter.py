from dataclasses import dataclass


@dataclass
class LearnedBloomFilter:
    model_threshold: float

    def query(self, score: float, backup_contains: bool) -> bool:
        if score >= self.model_threshold:
            return True
        return backup_contains
