from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = ROOT / "src" / "python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from learned_bloom.bloom_filter import BloomFilter
from learned_bloom.fused_scorer_py import FusedTrigramScorerPy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build backup Bloom filter from model false negatives")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/urldata.csv/urldata.csv"),
        help="Input CSV with url and label columns",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/fused_model"),
        help="Directory containing exported fused model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/learned_bloom_filter"),
        help="Directory for backup Bloom filter and metadata",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Target FPR for backup Bloom filter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def label_to_binary(label: str) -> int:
    return 1 if str(label).strip().lower() in {"bad", "malicious", "1", "true"} else 0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'url' and 'label' columns")

    urls = df["url"].astype(str).to_numpy()
    labels = df["label"].map(label_to_binary).astype(np.int32).to_numpy()

    positive_urls = urls[labels == 1]
    print(f"[learned-bf] Total URLs: {len(urls)}, Positives: {len(positive_urls)}")

    metadata_path = args.model_dir / "metadata.json"
    weights_path = args.model_dir / "weights.bin"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        import learned_bloom_cpp as cpp  # type: ignore
        scorer = cpp.FusedTrigramScorer(str(weights_path))
        print("[learned-bf] Using C++ scorer")
    except ImportError:
        scorer = FusedTrigramScorerPy(str(weights_path))
        print("[learned-bf] Using pure Python scorer")

    threshold = float(metadata["threshold"])

    probs = []
    false_negatives = []
    for url in positive_urls:
        score = float(scorer.score(url))
        prob = 1.0 / (1.0 + np.exp(-score))
        probs.append(prob)

        if prob < threshold:
            false_negatives.append(url)

    false_negative_rate = len(false_negatives) / len(positive_urls) if positive_urls.size > 0 else 0.0
    print(
        f"[learned-bf] False negatives: {len(false_negatives)} / {len(positive_urls)} "
        f"({false_negative_rate * 100:.2f}%)"
    )

    if len(false_negatives) == 0:
        print("[learned-bf] No false negatives, backup filter not needed")
        backup_bf = None
        backup_bf_memory = 0
    else:
        backup_bf = BloomFilter(expected_items=len(false_negatives), target_fpr=args.target_fpr)
        backup_bf.bulk_add(false_negatives)
        backup_bf_memory = backup_bf.memory_bytes
        print(
            f"[learned-bf] Backup Bloom filter: {backup_bf_memory} bytes "
            f"(estimated FPR: {backup_bf.estimated_fpr():.6f})"
        )

    model_size = weights_path.stat().st_size
    total_memory = model_size + backup_bf_memory

    summary = {
        "input": str(args.input),
        "positive_urls": int(len(positive_urls)),
        "false_negatives": int(len(false_negatives)),
        "false_negative_rate": float(false_negative_rate),
        "model_size_bytes": model_size,
        "backup_bloom_memory_bytes": backup_bf_memory,
        "total_memory_bytes": total_memory,
        "backup_bloom_target_fpr": args.target_fpr,
        "backup_bloom_estimated_fpr": float(backup_bf.estimated_fpr() if backup_bf else 0.0),
        "backup_bloom_num_bits": int(backup_bf.num_bits) if backup_bf else 0,
        "backup_bloom_num_hashes": int(backup_bf.num_hashes) if backup_bf else 0,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[learned-bf] Saved summary to: {args.output_dir / 'summary.json'}")
    print(f"[learned-bf] Model size: {model_size} bytes")
    print(f"[learned-bf] Total learned BF size: {total_memory} bytes")


if __name__ == "__main__":
    main()
