import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all three steps comparison")
    parser.add_argument(
        "--hash-benchmark",
        type=Path,
        default=Path("results/hash_benchmark/summary.json"),
        help="Path to hash benchmark JSON",
    )
    parser.add_argument(
        "--fused-benchmark",
        type=Path,
        default=Path("results/fused_cpp_benchmark/summary.json"),
        help="Path to fused benchmark JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/step_01_02_03_comparison.png"),
        help="Path to output PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.hash_benchmark.open("r", encoding="utf-8") as f:
        hash_report = json.load(f)

    with args.fused_benchmark.open("r", encoding="utf-8") as f:
        fused_report = json.load(f)

    steps = ["Step 01\n(Word Tokens)", "Step 02\n(Char Trigrams)", "Step 03\n(Fused Py)"]
    mean_latencies = [
        hash_report["results"][0]["mean_latency_ns"],
        hash_report["results"][1]["mean_latency_ns"],
        fused_report["mean_latency_ns"],
    ]
    qps_values = [
        hash_report["results"][0]["qps"],
        hash_report["results"][1]["qps"],
        fused_report["qps"],
    ]
    auc_values = [
        hash_report["results"][0]["roc_auc"],
        hash_report["results"][1]["roc_auc"],
        fused_report["roc_auc"],
    ]

    speedup_vs_01 = np.array(mean_latencies[0]) / np.array(mean_latencies)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Learned Bloom Filter: Step 01 vs 02 vs 03", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    colors = ["#2f6fed", "#18a999", "#ff6b35"]
    bars = ax.bar(steps, mean_latencies, color=colors)
    ax.set_title("Mean Latency (ns/query)")
    ax.set_ylabel("Latency (ns)")
    for i, (bar, val) in enumerate(zip(bars, mean_latencies)):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    bars = ax.bar(steps, qps_values, color=colors)
    ax.set_title("Throughput (queries/sec)")
    ax.set_ylabel("QPS")
    for i, (bar, val) in enumerate(zip(bars, qps_values)):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    ax.plot(steps, speedup_vs_01, marker="o", linewidth=2, markersize=8, color="#ff6b35", label="Speedup vs Step 01")
    ax.set_title("Speedup vs Step 01")
    ax.set_ylabel("Speedup factor")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(speedup_vs_01) * 1.1])
    for i, val in enumerate(speedup_vs_01):
        ax.text(i, val + 0.1, f"{val:.1f}x", ha="center", fontsize=9)

    ax = axes[1, 1]
    bars = ax.bar(steps, auc_values, color=colors)
    ax.set_title("Model Quality (ROC-AUC)")
    ax.set_ylabel("AUC")
    ax.set_ylim([0.98, 1.0])
    for i, (bar, val) in enumerate(zip(bars, auc_values)):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote comparison plot to: {args.output}")
    print(f"\nStep 03 vs Step 01 speedup: {speedup_vs_01[2]:.1f}x")
    print(f"Step 03 vs Step 02 speedup: {speedup_vs_01[1] / speedup_vs_01[2]:.1f}x")


if __name__ == "__main__":
    main()
