import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learned vs standard Bloom filter comparison")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/learned_vs_standard_benchmark/summary.json"),
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/learned_vs_standard_comparison.png"),
        help="Path to output PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        report = json.load(f)

    methods = ["Standard\nBloom Filter", "Learned\nBloom Filter"]
    colors = ["#2f6fed", "#ff6b35"]

    standard = report["standard_bloom"]
    learned = report["learned_bloom"]

    fpr_values = [standard["measured_fpr"], learned["measured_fpr"]]
    latency_values = [standard["mean_latency_ns"], learned["mean_latency_ns"]]
    memory_values = [standard["memory_bytes"], learned["total_memory_bytes"]]
    qps_values = [standard["qps"], learned["qps"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Learned Bloom Filter vs Standard Bloom Filter", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    bars = ax.bar(methods, memory_values, color=colors)
    ax.set_title("Memory Footprint (bytes)")
    ax.set_ylabel("Memory (bytes)")
    ax.set_yscale("log")
    for bar, val in zip(bars, memory_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    bars = ax.bar(methods, fpr_values, color=colors)
    ax.set_title("False Positive Rate (measured)")
    ax.set_ylabel("FPR")
    for bar, val in zip(bars, fpr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.6f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    bars = ax.bar(methods, latency_values, color=colors)
    ax.set_title("Query Latency (ns)")
    ax.set_ylabel("Latency (ns)")
    for bar, val in zip(bars, latency_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    bars = ax.bar(methods, qps_values, color=colors)
    ax.set_title("Throughput (QPS)")
    ax.set_ylabel("QPS")
    for bar, val in zip(bars, qps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote comparison plot to: {args.output}")

    print(f"\nKey Metrics:")
    print(f"Standard BF memory: {standard['memory_bytes']:,} bytes")
    print(f"Learned BF memory: {learned['total_memory_bytes']:,} bytes")
    print(f"  - Model size: {learned['model_memory_bytes']:,} bytes")
    print(f"  - Backup Bloom: {learned['backup_bloom_memory_bytes']:,} bytes")
    print(f"\nStandard BF FPR: {standard['measured_fpr']:.6f}")
    print(f"Learned BF FPR: {learned['measured_fpr']:.6f}")
    print(f"\nStandard BF latency: {standard['mean_latency_ns']:.1f} ns")
    print(f"Learned BF latency: {learned['mean_latency_ns']:.1f} ns")
    print(f"Speedup: {standard['mean_latency_ns'] / learned['mean_latency_ns']:.2f}x")


if __name__ == "__main__":
    main()
