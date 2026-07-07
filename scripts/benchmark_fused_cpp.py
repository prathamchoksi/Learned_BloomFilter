from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = ROOT / "src" / "python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from learned_bloom.fused_trigram import build_trigram_matrix
from learned_bloom.fused_scorer_py import FusedTrigramScorerPy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fused trigram scorer")

    parser.add_argument(
        "--impl",
        choices=["python", "cpp"],
        default="cpp",
        help="Implementation to benchmark",
    )

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
        "--output",
        type=Path,
        default=None,
        help="Output JSON path",
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=20000,
        help="Number of per-query latency measurements",
    )

    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=500,
        help="Warmup query count before timing",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def label_to_binary(label: str) -> int:
    return 1 if str(label).strip().lower() in {
        "bad",
        "malicious",
        "1",
        "true",
    } else 0


def main() -> None:
    args = parse_args()

    if args.output is None:
        args.output = Path(
            f"results/fused_{args.impl}_benchmark/summary.json"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Benchmarking implementation : {args.impl.upper()}")
    print("=" * 60)

    df = pd.read_csv(args.input)

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Input CSV must contain 'url' and 'label' columns"
        )

    urls = df["url"].astype(str).to_numpy()
    labels = df["label"].map(label_to_binary).astype(np.int32).to_numpy()

    train_urls, test_urls, train_labels, test_labels = train_test_split(
        urls,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    metadata_path = args.model_dir / "metadata.json"
    weights_path = args.model_dir / "weights.bin"

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if args.impl == "python":
        print("[info] Using Python scorer")
        scorer = FusedTrigramScorerPy(str(weights_path))

    else:
        import learned_bloom_cpp as cpp

        print("[info] Using C++ scorer")
        scorer = cpp.FusedTrigramScorer(str(weights_path))

    threshold = float(metadata["threshold"])
    n_features = int(metadata["n_features"])

    sample_count = min(5000, len(test_urls))
    x_sample = build_trigram_matrix(test_urls[:sample_count], n_features)
    avg_nnz = float(np.asarray(x_sample.getnnz(axis=1)).mean())

    probs = []
    preds = []

    for url in test_urls:
        score = float(scorer.score(url))
        prob = float(1.0 / (1.0 + np.exp(-score)))
        probs.append(prob)
        preds.append(1 if prob >= threshold else 0)

    accuracy = float(
        accuracy_score(
            test_labels,
            np.asarray(preds, dtype=np.int32),
        )
    )

    roc_auc = float(
        roc_auc_score(
            test_labels,
            np.asarray(probs, dtype=np.float64),
        )
    )

    rng = np.random.default_rng(args.seed)

    idx = rng.integers(
        0,
        len(test_urls),
        size=args.num_queries + args.warmup_queries,
    )

    query_urls = test_urls[idx]

    print("[info] Warmup...")

    for i in range(args.warmup_queries):
        scorer.score(query_urls[i])

    print("[info] Benchmarking...")

    latencies_ns = []

    for i in range(
        args.warmup_queries,
        args.warmup_queries + args.num_queries,
    ):
        start = time.perf_counter_ns()
        scorer.score(query_urls[i])
        end = time.perf_counter_ns()

        latencies_ns.append(end - start)

    arr = np.asarray(latencies_ns, dtype=np.float64)

    mean_ns = float(arr.mean())
    p50_ns = float(np.percentile(arr, 50))
    p95_ns = float(np.percentile(arr, 95))
    qps = float(1e9 / mean_ns)

    summary = {
        "implementation": args.impl,
        "input": str(args.input),
        "model_dir": str(args.model_dir),
        "train_size": int(len(train_urls)),
        "test_size": int(len(test_urls)),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "avg_nnz_per_url": avg_nnz,
        "mean_latency_ns": mean_ns,
        "p50_latency_ns": p50_ns,
        "p95_latency_ns": p95_ns,
        "qps": qps,
        "num_queries": args.num_queries,
        "threshold": threshold,
        "n_features": n_features,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"Implementation : {args.impl.upper()}")
    print(f"Mean latency  : {mean_ns:.1f} ns")
    print(f"P50 latency   : {p50_ns:.1f} ns")
    print(f"P95 latency   : {p95_ns:.1f} ns")
    print(f"Throughput    : {qps:.0f} QPS")
    print(f"AUC           : {roc_auc:.4f}")
    print("=" * 60)

    print(f"Summary saved to : {args.output}")


if __name__ == "__main__":
    main() 