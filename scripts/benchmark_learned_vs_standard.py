from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser = argparse.ArgumentParser(description="Benchmark learned Bloom filter vs standard Bloom filter")
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
        "--learned-bf-dir",
        type=Path,
        default=Path("results/learned_bloom_filter"),
        help="Directory containing learned BF metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/learned_vs_standard_benchmark/summary.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Target FPR for both filters",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20000,
        help="Number of benchmark queries",
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
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'url' and 'label' columns")

    urls = df["url"].astype(str).to_numpy()
    labels = df["label"].map(label_to_binary).astype(np.int32).to_numpy()

    positive_urls = urls[labels == 1]
    negative_urls = urls[labels == 0]

    print(f"[benchmark] Positives: {len(positive_urls)}, Negatives: {len(negative_urls)}")

    standard_bf = BloomFilter(expected_items=len(positive_urls), target_fpr=args.target_fpr)
    standard_bf.bulk_add(positive_urls)

    metadata_path = args.model_dir / "metadata.json"
    weights_path = args.model_dir / "weights.bin"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        import learned_bloom_cpp as cpp  # type: ignore
        scorer = cpp.FusedTrigramScorer(str(weights_path))
    except ImportError:
        scorer = FusedTrigramScorerPy(str(weights_path))

    threshold = float(metadata["threshold"])

    false_negatives = []
    for url in positive_urls:
        score = float(scorer.score(url))
        prob = 1.0 / (1.0 + np.exp(-score))
        if prob < threshold:
            false_negatives.append(url)

    backup_bf = None
    if len(false_negatives) > 0:
        backup_bf = BloomFilter(expected_items=len(false_negatives), target_fpr=args.target_fpr)
        backup_bf.bulk_add(false_negatives)

    print(f"[benchmark] Standard BF memory: {standard_bf.memory_bytes} bytes")
    print(f"[benchmark] Learned BF model size: {weights_path.stat().st_size} bytes")
    if backup_bf:
        print(f"[benchmark] Learned BF backup size: {backup_bf.memory_bytes} bytes")

    rng = np.random.default_rng(args.seed)
    test_positives = rng.choice(positive_urls, size=min(len(positive_urls), 10000), replace=False)
    test_negatives = rng.choice(negative_urls, size=min(len(negative_urls), 10000), replace=False)
    test_urls = np.concatenate([test_positives, test_negatives])
    test_labels = np.concatenate([np.ones(len(test_positives)), np.zeros(len(test_negatives))])

    standard_pred = np.array([1 if url in standard_bf else 0 for url in test_urls], dtype=np.int32)
    standard_acc = accuracy_score(test_labels, standard_pred)
    standard_false_positives = np.sum((standard_pred == 1) & (test_labels == 0))
    standard_fpr = float(standard_false_positives / np.sum(test_labels == 0)) if np.sum(test_labels == 0) > 0 else 0.0

    learned_pred = []
    for url in test_urls:
        score = float(scorer.score(url))
        prob = 1.0 / (1.0 + np.exp(-score))
        if prob >= threshold:
            pred = 1
        else:
            pred = 1 if (backup_bf and url in backup_bf) else 0
        learned_pred.append(pred)
    learned_pred = np.array(learned_pred, dtype=np.int32)
    learned_acc = accuracy_score(test_labels, learned_pred)
    learned_false_positives = np.sum((learned_pred == 1) & (test_labels == 0))
    learned_fpr = float(learned_false_positives / np.sum(test_labels == 0)) if np.sum(test_labels == 0) > 0 else 0.0

    query_urls_idx = rng.integers(0, len(test_urls), size=args.num_queries)
    query_urls = test_urls[query_urls_idx]

    standard_latencies = []
    for url in query_urls:
        start = time.perf_counter_ns()
        _ = url in standard_bf
        end = time.perf_counter_ns()
        standard_latencies.append(end - start)

    learned_latencies = []
    for url in query_urls:
        start = time.perf_counter_ns()
        score = float(scorer.score(url))
        prob = 1.0 / (1.0 + np.exp(-score))
        if prob < threshold and backup_bf:
            _ = url in backup_bf
        end = time.perf_counter_ns()
        learned_latencies.append(end - start)

    standard_latencies = np.array(standard_latencies, dtype=np.float64)
    learned_latencies = np.array(learned_latencies, dtype=np.float64)

    standard_qps = 1e9 / standard_latencies.mean() if standard_latencies.mean() > 0 else 0.0
    learned_qps = 1e9 / learned_latencies.mean() if learned_latencies.mean() > 0 else 0.0

    learned_total_memory = weights_path.stat().st_size + (backup_bf.memory_bytes if backup_bf else 0)
    memory_reduction = (1.0 - learned_total_memory / standard_bf.memory_bytes) * 100

    report = {
        "positive_count": int(len(positive_urls)),
        "negative_count": int(len(negative_urls)),
        "test_positive_count": int(len(test_positives)),
        "test_negative_count": int(len(test_negatives)),
        "standard_bloom": {
            "memory_bytes": standard_bf.memory_bytes,
            "num_bits": standard_bf.num_bits,
            "num_hashes": standard_bf.num_hashes,
            "estimated_fpr": float(standard_bf.estimated_fpr()),
            "accuracy": float(standard_acc),
            "measured_fpr": standard_fpr,
            "mean_latency_ns": float(standard_latencies.mean()),
            "p50_latency_ns": float(np.percentile(standard_latencies, 50)),
            "p95_latency_ns": float(np.percentile(standard_latencies, 95)),
            "qps": standard_qps,
        },
        "learned_bloom": {
            "model_memory_bytes": weights_path.stat().st_size,
            "backup_bloom_memory_bytes": backup_bf.memory_bytes if backup_bf else 0,
            "total_memory_bytes": learned_total_memory,
            "false_negatives_count": int(len(false_negatives)) if backup_bf else 0,
            "accuracy": float(learned_acc),
            "measured_fpr": learned_fpr,
            "mean_latency_ns": float(learned_latencies.mean()),
            "p50_latency_ns": float(np.percentile(learned_latencies, 50)),
            "p95_latency_ns": float(np.percentile(learned_latencies, 95)),
            "qps": learned_qps,
        },
        "comparison": {
            "memory_reduction_percent": memory_reduction,
            "fpr_improvement_factor": standard_fpr / learned_fpr if learned_fpr > 0 else 0.0,
            "latency_ratio": learned_latencies.mean() / standard_latencies.mean(),
            "qps_ratio": learned_qps / standard_qps,
        },
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[benchmark] Wrote report to: {args.output}")
    print(f"[benchmark] Standard BF memory: {report['standard_bloom']['memory_bytes']} bytes")
    print(f"[benchmark] Learned BF memory: {report['learned_bloom']['total_memory_bytes']} bytes")
    print(f"[benchmark] Memory reduction: {memory_reduction:.1f}%")
    print(f"[benchmark] Standard BF FPR: {report['standard_bloom']['measured_fpr']:.6f}")
    print(f"[benchmark] Learned BF FPR: {report['learned_bloom']['measured_fpr']:.6f}")


if __name__ == "__main__":
    main()
