
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend import COMPILATION_RECORDS, make_cpp_backend
from model import TinyModel


class MultiplyAdd(nn.Module):
    def forward(self, x):
        return x * 2 + 3


class AddRelu(nn.Module):
    def forward(self, x):
        return torch.relu(x + 3)


class LongChain(nn.Module):
    def forward(self, x):
        x = x * 2
        x = x + 3
        x = torch.relu(x)
        x = x * 0.5
        return x + 1


WORKLOADS = {
    "multiply_add": MultiplyAdd,
    "add_relu": AddRelu,
    "multiply_add_relu": TinyModel,
    "long_chain": LongChain,
}

SIZES = [1_000, 10_000, 100_000, 1_000_000]
WARMUP = 10
RUNS = 100


def measure(fn, x):
    for _ in range(WARMUP):
        fn(x)

    samples = []

    for _ in range(RUNS):
        start = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - start) * 1000)

    return statistics.median(samples)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)

    results = []
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("Fusion benchmark")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Warmups: {WARMUP}, measured runs: {RUNS}")

    for name, model_class in WORKLOADS.items():
        for size in SIZES:
            x = torch.randn(size, dtype=torch.float32)
            model = model_class().eval()
            expected = model(x)

            latencies = {}
            compile_times = {}

            for fuse in (False, True):
                mode = "fused" if fuse else "unfused"

                torch._dynamo.reset()
                COMPILATION_RECORDS.clear()

                compiled = torch.compile(
                    model,
                    backend=make_cpp_backend(fuse=fuse),
                    dynamic=False,
                )

                # First execution triggers compilation.
                actual = compiled(x)

                torch.testing.assert_close(
                    actual, expected, rtol=1e-5, atol=1e-6
                )

                compile_times[mode] = sum(
                    record["compilation_ms"]
                    for record in COMPILATION_RECORDS
                )

                latencies[mode] = measure(compiled, x)

            speedup = latencies["unfused"] / latencies["fused"]

            row = {
                "workload": name,
                "tensor_size": size,
                "unfused_ms": latencies["unfused"],
                "fused_ms": latencies["fused"],
                "speedup": speedup,
                "unfused_cpp_compile_ms": compile_times["unfused"],
                "fused_cpp_compile_ms": compile_times["fused"],
            }

            results.append(row)

            print(
                f"{name:20s} size={size:>8,} "
                f"unfused={latencies['unfused']:.4f} ms "
                f"fused={latencies['fused']:.4f} ms "
                f"speedup={speedup:.2f}x"
            )

    output_path = results_dir / "fusion_results.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
