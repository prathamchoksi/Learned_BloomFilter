from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = ROOT / "src" / "python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from learned_bloom.bloom_filter import BloomFilter  # noqa: E402
from learned_bloom.fused_trigram import build_trigram_matrix  # noqa: E402


@dataclass(frozen=True)
class VariantConfig:
    name: str
    kind: str
    analyzer: str | None = None
    ngram_range: tuple[int, int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark memory frontier for learned BF variants")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/urldata.csv/urldata.csv"),
        help="Input CSV with url and label columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/memory_usage_variants/summary.json"),
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--target-fprs",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.05, 0.1],
        help="Target system FPR values",
    )
    parser.add_argument(
        "--n-features-cpp",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
        help="Candidate n_features for C++ trigram variant",
    )
    parser.add_argument(
        "--n-features-char",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
        help="Candidate n_features for Python char trigram variant",
    )
    parser.add_argument(
        "--n-features-word",
        type=int,
        nargs="+",
        default=[4096, 16384, 65536],
        help="Candidate n_features for Python word variant",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        help="Candidate model thresholds",
    )
    parser.add_argument(
        "--backup-fprs",
        type=float,
        nargs="+",
        default=[0.001, 0.01],
        help="Candidate backup Bloom filter FPR values",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total rows for faster runs (0 = use full dataset)",
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


def train_hashing_model(
    train_urls: np.ndarray,
    train_labels: np.ndarray,
    n_features: int,
    analyzer: str,
    ngram_range: tuple[int, int],
    seed: int,
) -> tuple[HashingVectorizer, SGDClassifier]:
    vectorizer = HashingVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=True,
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-6,
        max_iter=20,
        tol=1e-3,
        random_state=seed,
    )
    x_train = vectorizer.transform(train_urls)
    model.fit(x_train, train_labels)
    return vectorizer, model


def train_fused_trigram_model(
    train_urls: np.ndarray,
    train_labels: np.ndarray,
    n_features: int,
    seed: int,
) -> SGDClassifier:
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


def score_hashing_probs(model: SGDClassifier, vectorizer: HashingVectorizer, urls: np.ndarray) -> np.ndarray:
    x = vectorizer.transform(urls)
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def score_fused_probs(model: SGDClassifier, urls: np.ndarray, n_features: int) -> np.ndarray:
    x = build_trigram_matrix(urls, n_features)
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def fused_model_size_bytes(n_features: int) -> int:
    # weights.bin format: uint64 n_features + float64 intercept + float64[n_features] weights
    return 8 + 8 + 8 * n_features


def sklearn_linear_model_size_bytes(model: SGDClassifier, n_features: int) -> int:
    coef_bytes = int(model.coef_.nbytes)
    intercept_bytes = int(model.intercept_.nbytes)
    # HashingVectorizer is stateless, so model arrays dominate explicit learned parameters.
    return max(coef_bytes + intercept_bytes, 8 * (n_features + 1))


def evaluate_configuration(
    positive_probs: np.ndarray,
    negative_probs: np.ndarray,
    threshold: float,
    backup_fpr: float,
    model_size_bytes: int,
) -> dict:
    fn_count = int(np.sum(positive_probs < threshold))
    model_fp_rate = float(np.mean(negative_probs >= threshold))

    if fn_count == 0:
        backup_memory = 0
        backup_estimated_fpr = 0.0
    else:
        backup = BloomFilter(expected_items=fn_count, target_fpr=backup_fpr)
        backup_memory = backup.memory_bytes
        backup_estimated_fpr = float(backup.estimated_fpr())

    system_fpr = model_fp_rate + (1.0 - model_fp_rate) * backup_estimated_fpr
    total_memory = model_size_bytes + backup_memory

    return {
        "threshold": threshold,
        "backup_fpr": backup_fpr,
        "false_negatives": fn_count,
        "model_fp_rate": model_fp_rate,
        "system_fpr": float(system_fpr),
        "model_size_bytes": int(model_size_bytes),
        "backup_bloom_memory_bytes": int(backup_memory),
        "total_memory_bytes": int(total_memory),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'url' and 'label' columns")

    if args.max_samples > 0 and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)

    urls = df["url"].astype(str).to_numpy()
    labels = df["label"].map(label_to_binary).astype(np.int32).to_numpy()
    positive_urls = urls[labels == 1]
    negative_urls = urls[labels == 0]

    train_urls, _, train_labels, _ = train_test_split(
        urls,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    print(f"[memory-benchmark] rows={len(urls)}, positives={len(positive_urls)}, negatives={len(negative_urls)}")

    variants = [
        (
            VariantConfig(name="cpp_trigram", kind="fused_trigram"),
            list(args.n_features_cpp),
        ),
        (
            VariantConfig(name="char_trigram_python", kind="hashing", analyzer="char", ngram_range=(3, 3)),
            list(args.n_features_char),
        ),
        (
            VariantConfig(name="word_default_python", kind="hashing", analyzer="word", ngram_range=(1, 1)),
            list(args.n_features_word),
        ),
    ]

    per_variant_feature_results: dict[str, dict[int, dict]] = {}

    for variant, feature_grid in variants:
        per_variant_feature_results[variant.name] = {}
        for n_features in feature_grid:
            print(f"[memory-benchmark] training {variant.name} n_features={n_features}")
            if variant.kind == "fused_trigram":
                model = train_fused_trigram_model(train_urls, train_labels, n_features, args.seed)
                pos_probs = score_fused_probs(model, positive_urls, n_features)
                neg_probs = score_fused_probs(model, negative_urls, n_features)
                model_size = fused_model_size_bytes(n_features)
            else:
                vectorizer, model = train_hashing_model(
                    train_urls,
                    train_labels,
                    n_features,
                    variant.analyzer or "word",
                    variant.ngram_range or (1, 1),
                    args.seed,
                )
                pos_probs = score_hashing_probs(model, vectorizer, positive_urls)
                neg_probs = score_hashing_probs(model, vectorizer, negative_urls)
                model_size = sklearn_linear_model_size_bytes(model, n_features)

            per_variant_feature_results[variant.name][n_features] = {
                "positive_probs": pos_probs,
                "negative_probs": neg_probs,
                "model_size_bytes": model_size,
            }

    target_results: dict[str, dict] = {}
    for target_fpr in args.target_fprs:
        standard_bf = BloomFilter(expected_items=len(positive_urls), target_fpr=target_fpr)
        standard_memory = standard_bf.memory_bytes
        t_key = str(target_fpr)
        target_results[t_key] = {
            "target_fpr": target_fpr,
            "standard_bloom": {
                "memory_bytes": int(standard_memory),
                "num_bits": int(standard_bf.num_bits),
                "num_hashes": int(standard_bf.num_hashes),
                "estimated_fpr": float(standard_bf.estimated_fpr()),
            },
            "best_variants": {},
        }

        for variant, feature_grid in variants:
            candidates = []
            for n_features in feature_grid:
                cached = per_variant_feature_results[variant.name][n_features]
                pos_probs = cached["positive_probs"]
                neg_probs = cached["negative_probs"]
                model_size = cached["model_size_bytes"]
                for threshold in args.thresholds:
                    for backup_fpr in args.backup_fprs:
                        cfg = evaluate_configuration(
                            pos_probs,
                            neg_probs,
                            threshold,
                            backup_fpr,
                            model_size,
                        )
                        cfg["n_features"] = int(n_features)
                        candidates.append(cfg)

            valid = [c for c in candidates if c["system_fpr"] <= target_fpr]
            if not valid:
                best = min(candidates, key=lambda x: x["system_fpr"])
                best["meets_target"] = False
            else:
                best = min(valid, key=lambda x: x["total_memory_bytes"])
                best["meets_target"] = True

            best["memory_mb"] = float(best["total_memory_bytes"] / (1024 * 1024))
            best["standard_memory_mb"] = float(standard_memory / (1024 * 1024))
            best["memory_reduction_percent"] = float(
                100.0 * (1.0 - best["total_memory_bytes"] / standard_memory)
            )

            target_results[t_key]["best_variants"][variant.name] = best

    report = {
        "input": str(args.input),
        "rows": int(len(urls)),
        "positive_count": int(len(positive_urls)),
        "negative_count": int(len(negative_urls)),
        "train_count": int(len(train_urls)),
        "seed": int(args.seed),
        "target_fprs": [float(x) for x in args.target_fprs],
        "threshold_grid": [float(x) for x in args.thresholds],
        "backup_fpr_grid": [float(x) for x in args.backup_fprs],
        "n_features": {
            "cpp_trigram": [int(x) for x in args.n_features_cpp],
            "char_trigram_python": [int(x) for x in args.n_features_char],
            "word_default_python": [int(x) for x in args.n_features_word],
        },
        "results_by_target_fpr": target_results,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[memory-benchmark] Wrote summary: {args.output}")
    for t in args.target_fprs:
        t_key = str(t)
        std_mem = target_results[t_key]["standard_bloom"]["memory_bytes"]
        cpp_best = target_results[t_key]["best_variants"]["cpp_trigram"]
        print(
            f"  target_fpr={t:.4f} std={std_mem}B cpp={cpp_best['total_memory_bytes']}B "
            f"reduction={cpp_best['memory_reduction_percent']:.1f}%"
        )


if __name__ == "__main__":
    main()
