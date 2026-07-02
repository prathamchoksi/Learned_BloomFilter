import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 01/02 hashing benchmark")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/urldata.csv/urldata.csv"),
        help="Input CSV with url and label columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hash_benchmark/summary.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=2**18,
        help="HashingVectorizer feature space",
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
    return 1 if str(label).strip().lower() in {"bad", "malicious", "1", "true"} else 0


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'url' and 'label' columns")

    urls = df["url"].astype(str).to_numpy()
    labels = df["label"].map(label_to_binary).astype(np.int32).to_numpy()
    return urls, labels


def benchmark_pipeline(
    name: str,
    analyzer: str,
    ngram_range: tuple[int, int],
    n_features: int,
    train_urls: np.ndarray,
    train_labels: np.ndarray,
    test_urls: np.ndarray,
    test_labels: np.ndarray,
    num_queries: int,
    warmup_queries: int,
    seed: int,
) -> dict:
    vectorizer = HashingVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=True,
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-6,
        max_iter=20,
        tol=1e-3,
        random_state=seed,
    )

    x_train = vectorizer.transform(train_urls)
    clf.fit(x_train, train_labels)

    x_test = vectorizer.transform(test_urls)
    decision = clf.decision_function(x_test)
    probs = 1.0 / (1.0 + np.exp(-decision))
    pred = (probs >= 0.5).astype(np.int32)

    acc = float(accuracy_score(test_labels, pred))
    auc = float(roc_auc_score(test_labels, probs))

    sample_count = min(5000, len(test_urls))
    x_sample = vectorizer.transform(test_urls[:sample_count])
    avg_nnz = float(np.asarray(x_sample.getnnz(axis=1)).mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(test_urls), size=num_queries + warmup_queries)
    query_urls = test_urls[idx]

    for i in range(warmup_queries):
        q = query_urls[i]
        _ = clf.decision_function(vectorizer.transform([q]))[0]

    latencies_ns: list[int] = []
    for i in range(warmup_queries, warmup_queries + num_queries):
        q = query_urls[i]
        start = time.perf_counter_ns()
        _ = clf.decision_function(vectorizer.transform([q]))[0]
        end = time.perf_counter_ns()
        latencies_ns.append(end - start)

    arr = np.asarray(latencies_ns, dtype=np.float64)
    mean_ns = float(arr.mean())
    p50_ns = float(np.percentile(arr, 50))
    p95_ns = float(np.percentile(arr, 95))
    qps = float(1e9 / mean_ns) if mean_ns > 0 else 0.0

    return {
        "name": name,
        "analyzer": analyzer,
        "ngram_range": [ngram_range[0], ngram_range[1]],
        "n_features": n_features,
        "accuracy": acc,
        "roc_auc": auc,
        "avg_nnz_per_url": avg_nnz,
        "mean_latency_ns": mean_ns,
        "p50_latency_ns": p50_ns,
        "p95_latency_ns": p95_ns,
        "qps": qps,
        "num_queries": num_queries,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    urls, labels = load_dataset(args.input)
    train_urls, test_urls, train_labels, test_labels = train_test_split(
        urls,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    step_01 = benchmark_pipeline(
        name="step_01_word_tokens",
        analyzer="word",
        ngram_range=(1, 1),
        n_features=args.n_features,
        train_urls=train_urls,
        train_labels=train_labels,
        test_urls=test_urls,
        test_labels=test_labels,
        num_queries=args.num_queries,
        warmup_queries=args.warmup_queries,
        seed=args.seed,
    )
    step_02 = benchmark_pipeline(
        name="step_02_char_trigrams",
        analyzer="char",
        ngram_range=(3, 3),
        n_features=args.n_features,
        train_urls=train_urls,
        train_labels=train_labels,
        test_urls=test_urls,
        test_labels=test_labels,
        num_queries=args.num_queries,
        warmup_queries=args.warmup_queries,
        seed=args.seed,
    )

    report = {
        "input": str(args.input),
        "train_size": int(len(train_urls)),
        "test_size": int(len(test_urls)),
        "results": [step_01, step_02],
    }

    for item in report["results"]:
        step_output = args.output.parent / f"{item['name']}.json"
        with step_output.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote benchmark report to: {args.output}")
    for item in report["results"]:
        print(
            f"{item['name']}: mean={item['mean_latency_ns']:.1f} ns, "
            f"qps={item['qps']:.0f}, avg_nnz={item['avg_nnz_per_url']:.1f}, "
            f"auc={item['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
