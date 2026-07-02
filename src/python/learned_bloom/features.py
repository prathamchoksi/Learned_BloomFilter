from dataclasses import dataclass


@dataclass
class FeatureConfig:
    ngram_n: int = 3
    lowercase: bool = True


def normalize_url(url: str, lowercase: bool = True) -> str:
    text = url.strip()
    return text.lower() if lowercase else text
