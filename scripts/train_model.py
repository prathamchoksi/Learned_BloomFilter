from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = ROOT / "src" / "python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from learned_bloom.fused_trigram import (  # noqa: E402
    FusedTrigramModelMetadata,
    build_trigram_matrix,
    export_fused_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fused char-trigram logistic model")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/urldata.csv/urldata.csv"),
        help="Input CSV with url and label columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fused_model"),
        help="Directory for exported fused model",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=2**18,
        help="Number of hashed trigram features",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for prediction",
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

    x_train = build_trigram_matrix(train_urls, args.n_features)
    x_test = build_trigram_matrix(test_urls, args.n_features)

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-6,
        max_iter=20,
        tol=1e-3,
        random_state=args.seed,
    )
    clf.fit(x_train, train_labels)

    decision = clf.decision_function(x_test)
    probs = 1.0 / (1.0 + np.exp(-decision))
    pred = (probs >= args.threshold).astype(np.int32)

    accuracy = float(accuracy_score(test_labels, pred))
    roc_auc = float(roc_auc_score(test_labels, probs))

    metadata = FusedTrigramModelMetadata(
        n_features=args.n_features,
        analyzer="char",
        ngram_range=(3, 3),
        hash_scheme="rolling_poly64",
        threshold=args.threshold,
    )
    weights_path, metadata_path = export_fused_model(
        args.output_dir,
        clf.coef_.reshape(-1),
        float(clf.intercept_[0]),
        metadata,
    )

    summary = {
        "input": str(args.input),
        "train_size": int(len(train_urls)),
        "test_size": int(len(test_urls)),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "weights_path": str(weights_path),
        "metadata_path": str(metadata_path),
        "threshold": args.threshold,
        "n_features": args.n_features,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported fused model to: {args.output_dir}")
    print(f"accuracy={accuracy:.4f}, roc_auc={roc_auc:.4f}")


if __name__ == "__main__":
    main()
