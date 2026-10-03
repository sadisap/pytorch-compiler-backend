import csv
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SRC_DIR))

from backend import cpp_backend
from model import TinyModel


TENSOR_SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000,
]

WARMUP_RUNS = 10
MEASURED_RUNS = 100


def measure_latency(function, x):
    for _ in range(WARMUP_RUNS):
        function(x)

    times_ms = []

    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        function(x)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        times_ms.append(elapsed_ms)

    return statistics.median(times_ms)


def benchmark_size(size):
    print(f"\n{'=' * 60}")
    print(f"Tensor size: {size:,}")
    print(f"{'=' * 60}")

    x = torch.randn(
        size,
        dtype=torch.float32,
    )

    model = TinyModel()
    model.eval()

    # ---------------------------------------------------------
    # PyTorch eager
    # ---------------------------------------------------------

    with torch.no_grad():
        eager_output = model(x)

        eager_latency = measure_latency(
            model,
            x,
        )

    print(
        f"PyTorch eager:       "
        f"{eager_latency:.6f} ms"
    )

    # ---------------------------------------------------------
    # Our C++ backend
    #
    # Call once before timing so graph capture, C++ generation,
    # compilation, and library loading are not included in the
    # execution latency.
    # ---------------------------------------------------------

    torch._dynamo.reset()

    cpp_model = torch.compile(
        model,
        backend=cpp_backend,
        dynamic=False,
    )

    with torch.no_grad():
        cpp_output = cpp_model(x)

        if not torch.allclose(
            eager_output,
            cpp_output,
        ):
            raise RuntimeError(
                "C++ backend output does not match PyTorch."
            )

        cpp_latency = measure_latency(
            cpp_model,
            x,
        )

    print(
        f"Our C++ backend:     "
        f"{cpp_latency:.6f} ms"
    )

    # ---------------------------------------------------------
    # TorchInductor
    #
    # Again, compile before timing so compilation is not part
    # of execution latency.
    # ---------------------------------------------------------

    torch._dynamo.reset()

    inductor_model = torch.compile(
        model,
        backend="inductor",
        dynamic=False,
    )

    with torch.no_grad():
        inductor_output = inductor_model(x)

        if not torch.allclose(
            eager_output,
            inductor_output,
        ):
            raise RuntimeError(
                "TorchInductor output does not match PyTorch."
            )

        inductor_latency = measure_latency(
            inductor_model,
            x,
        )

    print(
        f"TorchInductor:       "
        f"{inductor_latency:.6f} ms"
    )

    print("Correctness:         PASS")

    return {
        "tensor_size": size,
        "eager_ms": eager_latency,
        "cpp_backend_ms": cpp_latency,
        "inductor_ms": inductor_latency,
    }


def save_results(results):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "baseline_results.csv"
    )

    with output_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tensor_size",
                "eager_ms",
                "cpp_backend_ms",
                "inductor_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )


def print_summary(results):
    print("\n")
    print("=" * 82)
    print("BASELINE BENCHMARK RESULTS")
    print("=" * 82)

    print(
        f"{'Size':>12} "
        f"{'Eager (ms)':>18} "
        f"{'Our Backend (ms)':>22} "
        f"{'Inductor (ms)':>20}"
    )

    print("-" * 82)

    for result in results:
        print(
            f"{result['tensor_size']:>12,} "
            f"{result['eager_ms']:>18.6f} "
            f"{result['cpp_backend_ms']:>22.6f} "
            f"{result['inductor_ms']:>20.6f}"
        )

    print("=" * 82)


def main():
    torch.manual_seed(0)

    print("PyTorch Compiler Backend")
    print("Baseline Benchmark")
    print()
    print(
        f"Warmup runs:   {WARMUP_RUNS}"
    )
    print(
        f"Measured runs: {MEASURED_RUNS}"
    )

    results = []

    for size in TENSOR_SIZES:
        result = benchmark_size(size)
        results.append(result)

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()