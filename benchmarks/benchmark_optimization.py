
import csv
import platform
import statistics
import time
from pathlib import Path

import torch

from src.backend import make_cpp_backend


class MultiplyAdd(torch.nn.Module):
    def forward(self, x):
        return x * 2 + 3


class AddRelu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 3)


class MultiplyAddRelu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2 + 3)


class LongChain(torch.nn.Module):
    def forward(self, x):
        x = x * 2
        x = x + 3
        x = torch.relu(x)
        x = x * 0.5
        return x + 1


WORKLOADS = {
    "multiply_add": MultiplyAdd,
    "add_relu": AddRelu,
    "multiply_add_relu": MultiplyAddRelu,
    "long_chain": LongChain,
}

SIZES = [1_000, 10_000, 100_000, 1_000_000]
OPT_LEVELS = ["O0", "O2", "O3"]
REPEATS = 5
WARMUPS = 10
ITERATIONS = 50

RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "optimization_results_run2.csv"
)


def measure(fn, x):
    for _ in range(WARMUPS):
        fn(x)

    samples = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        for _ in range(ITERATIONS):
            fn(x)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000 / ITERATIONS

        samples.append(elapsed_ms)

    return statistics.median(samples), statistics.stdev(samples)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    print("PyTorch:", torch.__version__)
    print("Platform:", platform.platform())
    print("Threads:", torch.get_num_threads())

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for workload_name, model_class in WORKLOADS.items():
        model = model_class().eval()

        for size in SIZES:
            x = torch.randn(size)
            expected = model(x)

            for opt_level in OPT_LEVELS:
                for fuse in (False, True):
                    torch._dynamo.reset()

                    compiled = torch.compile(
                        model,
                        backend=make_cpp_backend(
                            fuse=fuse,
                            opt_level=opt_level,
                        ),
                        fullgraph=True,
                    )

                    actual = compiled(x)

                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=1e-5,
                        atol=1e-6,
                    )

                    median_ms, stdev_ms = measure(
                        compiled, x
                    )

                    mode = "fused" if fuse else "unfused"

                    row = {
                        "workload": workload_name,
                        "size": size,
                        "opt_level": opt_level,
                        "mode": mode,
                        "median_ms": median_ms,
                        "stdev_ms": stdev_ms,
                    }

                    rows.append(row)

                    print(
                        f"{workload_name:20s} "
                        f"N={size:8d} "
                        f"{opt_level} "
                        f"{mode:7s} "
                        f"{median_ms:.5f} ms"
                    )

    with RESULTS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "workload",
                "size",
                "opt_level",
                "mode",
                "median_ms",
                "stdev_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
