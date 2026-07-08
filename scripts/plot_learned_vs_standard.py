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

    # Light theme colors
    colors = ["steelblue", "darkorange"]

    standard = report["standard_bloom"]
    learned = report["learned_bloom"]

    fpr_values = [standard["measured_fpr"], learned["measured_fpr"]]
    latency_values = [standard["mean_latency_ns"], learned["mean_latency_ns"]]
    memory_values = [standard["memory_bytes"], learned["total_memory_bytes"]]
    qps_values = [standard["qps"], learned["qps"]]

    plt.style.use("default")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    fig.patch.set_facecolor("white")

    for row in axes:
        for ax in row:
            ax.set_facecolor("white")

    fig.suptitle(
        "Learned Bloom Filter vs Standard Bloom Filter",
        fontsize=16,
        fontweight="bold",
    )

    # ---------------- Memory ----------------
    ax = axes[0, 0]

    bars = ax.bar(methods, memory_values, color=colors)

    ax.set_title("Memory Footprint")
    ax.set_ylabel("Memory (bytes)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, memory_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # ---------------- FPR ----------------
    ax = axes[0, 1]

    bars = ax.bar(methods, fpr_values, color=colors)

    ax.set_title("False Positive Rate")
    ax.set_ylabel("FPR")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, fpr_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # ---------------- Latency ----------------
    ax = axes[1, 0]

    bars = ax.bar(methods, latency_values, color=colors)

    ax.set_title("Query Latency")
    ax.set_ylabel("Latency (ns)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, latency_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # ---------------- Throughput ----------------
    ax = axes[1, 1]

    bars = ax.bar(methods, qps_values, color=colors)

    ax.set_title("Throughput")
    ax.set_ylabel("Queries / Second")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, qps_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)

    print(f"Wrote comparison plot to: {args.output}")

    print("\nKey Metrics:")
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