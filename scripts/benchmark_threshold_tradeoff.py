from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = ROOT / "src" / "python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from learned_bloom.bloom_filter import BloomFilter  # noqa: E402
from learned_bloom.fused_trigram import build_trigram_matrix  # noqa: E402


@dataclass(frozen=True)
class FeatureResult:
    n_features: int
    threshold: float
    system_fpr: float
    backup_fn_count: int
    backup_memory_bytes: int
    model_size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark threshold tradeoff for fused trigram learned BF")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/urldata.csv/urldata.csv"),
        help="Input CSV with url and label columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/threshold_tradeoff/summary.json"),
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("results/threshold_tradeoff/threshold_tradeoff.png"),
        help="Output chart path",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 16384],
        help="Feature grid to evaluate",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        help="Threshold grid to evaluate",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        nargs="+",
        default=[0.01, 0.05],
        help="Target FPR values to evaluate",
    )
    parser.add_argument(
        "--backup-fpr",
        type=float,
        default=0.01,
        help="Target FPR for the backup Bloom filter",
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


def fit_model(train_urls: np.ndarray, train_labels: np.ndarray, n_features: int, seed: int) -> SGDClassifier:
    x_train = build_trigram_matrix(train_urls, n_features)
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-6,
        max_iter=20,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(x_train, train_labels)
    return model


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'url' and 'label' columns")

    urls = df["url"].astype(str).to_numpy()
    labels = df["label"].map(label_to_binary).astype(np.int32).to_numpy()
    train_urls, test_urls, train_labels, test_labels = train_test_split(
        urls,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    train_positive_urls = train_urls[train_labels == 1]
    test_positive_urls = test_urls[test_labels == 1]
    test_negative_urls = test_urls[test_labels == 0]

    print(
        f"[tradeoff] train={len(train_urls)} test={len(test_urls)} "
        f"train_pos={len(train_positive_urls)} test_pos={len(test_positive_urls)} test_neg={len(test_negative_urls)}"
    )

    results_by_target_fpr: dict[str, dict[str, list[dict]]] = {}

    for target_fpr in args.target_fpr:
        print(f"[tradeoff] evaluating target_fpr={target_fpr:g}")
        target_results: dict[str, list[dict]] = {}
        for n_features in args.n_features:
            print(f"[tradeoff] fitting n_features={n_features}")
            model = fit_model(train_urls, train_labels, n_features, args.seed)

            train_pos_probs = sigmoid(model.decision_function(build_trigram_matrix(train_positive_urls, n_features)))
            test_neg_probs = sigmoid(model.decision_function(build_trigram_matrix(test_negative_urls, n_features)))

            model_size_bytes = 8 + 8 + 8 * n_features
            series: list[dict] = []

            standard_bf = BloomFilter(expected_items=len(train_positive_urls), target_fpr=target_fpr)
            standard_memory_bytes = standard_bf.memory_bytes

            for threshold in args.thresholds:
                backup_fn_count = int(np.sum(train_pos_probs < threshold))
                backup_memory_bytes = 0
                backup_estimated_fpr = 0.0
                if backup_fn_count > 0:
                    backup_bf = BloomFilter(expected_items=backup_fn_count, target_fpr=args.backup_fpr)
                    backup_memory_bytes = backup_bf.memory_bytes
                    backup_estimated_fpr = float(backup_bf.estimated_fpr())

                model_positive_rate = float(np.mean(test_neg_probs >= threshold))
                system_fpr = model_positive_rate + (1.0 - model_positive_rate) * backup_estimated_fpr
                total_memory_bytes = int(model_size_bytes + backup_memory_bytes)
                valid = system_fpr <= target_fpr
                memory_reduction_percent = None
                if valid:
                    memory_reduction_percent = float(100.0 * (1.0 - total_memory_bytes / standard_memory_bytes))

                series.append(
                    {
                        "n_features": int(n_features),
                        "threshold": float(threshold),
                        "system_fpr": float(system_fpr),
                        "valid": bool(valid),
                        "backup_fn_count": int(backup_fn_count),
                        "backup_memory_bytes": int(backup_memory_bytes),
                        "model_size_bytes": int(model_size_bytes),
                        "total_memory_bytes": total_memory_bytes,
                        "standard_memory_bytes": int(standard_memory_bytes),
                        "memory_reduction_percent": memory_reduction_percent,
                    }
                )

            target_results[str(n_features)] = series

        results_by_target_fpr[str(target_fpr)] = target_results

    report = {
        "input": str(args.input),
        "seed": int(args.seed),
        "target_fprs": [float(x) for x in args.target_fpr],
        "backup_fpr": float(args.backup_fpr),
        "n_features": [int(x) for x in args.n_features],
        "thresholds": [float(x) for x in args.thresholds],
        "results_by_target_fpr": results_by_target_fpr,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[tradeoff] Wrote summary to: {args.output}")


if __name__ == "__main__":
    main()
