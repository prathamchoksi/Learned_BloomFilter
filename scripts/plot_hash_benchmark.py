import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Step 01/02 benchmark results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/hash_benchmark/summary.json"),
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hash_benchmark/plots/hash_benchmark_plot.png"),
        help="Path to output PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        report = json.load(f)

    names = [x["name"] for x in report["results"]]
    labels = [name.replace("step_", "").replace("_", " ") for name in names]

    mean_latency_ns = np.array([x["mean_latency_ns"] for x in report["results"]], dtype=np.float64)
    qps = np.array([x["qps"] for x in report["results"]], dtype=np.float64)
    avg_nnz = np.array([x["avg_nnz_per_url"] for x in report["results"]], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("HashingVectorizer Step 01 vs Step 02", fontsize=13)

    axes[0].bar(labels, mean_latency_ns, color=["#2f6fed", "#18a999"])
    axes[0].set_title("Mean Latency (ns/query)")
    axes[0].tick_params(axis="x", labelrotation=10)

    axes[1].bar(labels, qps, color=["#2f6fed", "#18a999"])
    axes[1].set_title("Throughput (QPS)")
    axes[1].tick_params(axis="x", labelrotation=10)

    axes[2].bar(labels, avg_nnz, color=["#2f6fed", "#18a999"])
    axes[2].set_title("Avg NNZ per URL")
    axes[2].tick_params(axis="x", labelrotation=10)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote plot to: {args.output}")


if __name__ == "__main__":
    main()
