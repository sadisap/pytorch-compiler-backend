# Baseline Benchmark Results

These results measure the execution latency of the basic C++ backend before
the main compiler optimizations are added.

The benchmark compares:

- PyTorch eager execution
- the custom C++ backend
- TorchInductor

The test computation performs:

```text
input -> multiply by 2 -> add 3 -> ReLU -> output
```

## Method

The benchmark uses tensor sizes of:

- 1,000
- 10,000
- 100,000
- 1,000,000 elements

For each tensor size, each version is run 10 times for warmup before
measurement. The reported latency is the median of 100 measured runs.

Compilation is performed before the execution measurements and is not
included in the execution latency. This keeps the benchmark focused on
the time needed to execute the computation after compilation.

The output from the custom C++ backend and TorchInductor is also compared
with normal PyTorch output to check correctness before the timing result
is recorded.

## Baseline Results

| Tensor Size | PyTorch Eager (ms) | Custom C++ Backend (ms) | TorchInductor (ms) |
|---:|---:|---:|---:|
| 1,000 | 0.007833 | 0.085542 | 0.028791 |
| 10,000 | 0.012771 | 0.573896 | 0.056458 |
| 100,000 | 0.071813 | 5.491645 | 0.072604 |
| 1,000,000 | 0.171750 | 54.719833 | 0.127688 |

The custom C++ backend is slower than both PyTorch eager execution and
TorchInductor in these baseline measurements. The difference becomes
larger as the tensor size increases.

This is expected at this stage because the custom backend is intentionally
basic and has not been optimized yet. The generated C++ currently uses
separate loops and temporary vectors for the multiply, add, and ReLU
operations.

These measurements give the project a baseline that future optimization
experiments can be compared against.

## Files

The raw benchmark measurements are stored in:

```text
results/baseline_results.csv
```

The baseline latency plot is stored in:

```text
results/baseline_latency.png
```

The benchmark can be run with:

```bash
python benchmarks/benchmark.py
```

The plot can be regenerated from the saved CSV results with:

```bash
python benchmarks/plot_results.py
```

## Next Steps

The next stages of the project will use this baseline to study how
compiler optimizations affect execution performance. The main planned
optimizations are elementwise operator fusion and CPU vectorization.

The goal is not to make the custom backend outperform PyTorch or
TorchInductor. The goal is to measure how these individual optimization
decisions change the performance of the small custom backend.
```

Then just regenerate the graph from the final CSV:

```bash
python benchmarks/plot_results.py