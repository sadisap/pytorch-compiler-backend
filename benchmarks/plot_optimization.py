
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "optimization_results.csv"
OUTPUT_PATH = ROOT / "results" / "optimization_speedups.png"

with CSV_PATH.open() as f:
    rows = list(csv.DictReader(f))

workloads = [
    "multiply_add",
    "add_relu",
    "multiply_add_relu",
    "long_chain",
]

levels = ["O0", "O2", "O3"]
sizes = [1_000, 10_000, 100_000, 1_000_000]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, workload in zip(axes.flat, workloads):
    for level in levels:
        speedups = []

        for size in sizes:
            matching = {
                r["mode"]: float(r["median_ms"])
                for r in rows
                if r["workload"] == workload
                and int(r["size"]) == size
                and r["opt_level"] == level
            }

            speedups.append(
                matching["unfused"] / matching["fused"]
            )

        ax.plot(
            sizes,
            speedups,
            marker="o",
            label=f"-{level}",
        )

    ax.axhline(1, color="gray", linestyle="--")
    ax.set_xscale("log")
    ax.set_title(workload.replace("_", " ").title())
    ax.set_xlabel("Tensor elements")
    ax.set_ylabel("Fusion speedup (×)")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "Graph Fusion Speedup Across C++ Optimization Levels",
    fontsize=15,
)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=200)

print(f"Saved figure: {OUTPUT_PATH}")
