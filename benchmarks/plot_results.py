import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "baseline_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_latency.png"


def load_results():
    sizes = []
    eager = []
    cpp_backend = []
    inductor = []

    with RESULTS_PATH.open() as file:
        reader = csv.DictReader(file)

        for row in reader:
            sizes.append(int(row["tensor_size"]))
            eager.append(float(row["eager_ms"]))
            cpp_backend.append(float(row["cpp_backend_ms"]))
            inductor.append(float(row["inductor_ms"]))

    return sizes, eager, cpp_backend, inductor


def main():
    sizes, eager, cpp_backend, inductor = load_results()

    plt.figure(figsize=(8, 5))

    plt.plot(
        sizes,
        eager,
        marker="o",
        label="PyTorch Eager",
    )

    plt.plot(
        sizes,
        cpp_backend,
        marker="o",
        label="Our C++ Backend",
    )

    plt.plot(
        sizes,
        inductor,
        marker="o",
        label="TorchInductor",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Tensor Size")
    plt.ylabel("Median Execution Latency (ms)")
    plt.title("Baseline Execution Latency")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
    )

    print(f"Plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()