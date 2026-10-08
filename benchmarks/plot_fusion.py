
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "fusion_results.csv"
OUTPUT = ROOT / "results" / "fusion_speedup.png"


def main():
    groups = defaultdict(list)

    with INPUT.open() as file:
        for row in csv.DictReader(file):
            groups[row["workload"]].append(
                (int(row["tensor_size"]), float(row["speedup"]))
            )

    plt.figure(figsize=(9, 5))

    for workload, values in groups.items():
        values.sort()
        sizes, speedups = zip(*values)

        plt.plot(sizes, speedups, marker="o", label=workload)

    plt.axhline(1.0, linestyle="--", label="No speedup")
    plt.xscale("log")

    plt.xlabel("Tensor size (elements)")
    plt.ylabel("Unfused latency / fused latency")
    plt.title("Elementwise Fusion Speedup")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUTPUT, dpi=200)

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
