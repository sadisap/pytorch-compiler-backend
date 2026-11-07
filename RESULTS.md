
# Compiler Optimization and Operator Fusion Study

## Research Question

How do explicit operator fusion and C++ compiler optimization levels
affect execution latency and intermediate-memory requirements in a
small PyTorch CPU compiler backend?

## System

The backend uses PyTorch Dynamo/FX to capture supported computation
graphs, generates C++17 kernels, compiles them with a native C++
compiler, and executes them through ctypes.

Supported workloads consist of linear chains of scalar multiplication,
scalar addition, and ReLU operations on CPU float32 tensors.

## Experimental Design

Four workloads were evaluated:

- Multiply + Add
- Add + ReLU
- Multiply + Add + ReLU
- Long Chain (five operations)

Configurations:

- Tensor sizes: 1,000; 10,000; 100,000; 1,000,000
- Compiler optimization: O0, O2, O3
- Execution modes: fused and unfused

Total: 96 configurations per benchmark run.

Execution timing excluded compilation, used 10 warmup calls,
and reported the median of five batches of 50 calls.

Two benchmark runs were collected.

## Results

At one million elements:

| Workload | O0 Run 1 | O2 Run 1 | O3 Run 1 | O3 Run 2 |
|---|---:|---:|---:|---:|
| Multiply + Add | 20.83x | 2.84x | 3.67x | 2.75x |
| Add + ReLU | 6.36x | 3.20x | 3.12x | 2.20x |
| Multiply + Add + ReLU | 6.93x | 5.80x | 4.15x | 4.36x |
| Long Chain | 11.60x | 6.95x | 6.38x | 6.80x |

Speedup is calculated as unfused latency divided by fused latency.

Fusion consistently improved measured latency for these large-tensor
configurations, although some speedup estimates varied substantially
between runs.

## Compiler Analysis

Clang O3 vectorization reports confirmed:

- Fused long-chain kernel: one vectorized loop
- Unfused long-chain kernel: five vectorized loops
- Vectorization width: four float32 elements
- Interleaved count: four

Both implementations use SIMD. Fusion reduces separate processing
passes and intermediate-array requirements rather than simply
enabling vectorization.

## Intermediate Memory

At one million float32 elements:

| Workload | Unfused temporary storage | Fused |
|---|---:|---:|
| Multiply + Add | 7.63 MiB | 0 MiB |
| Add + ReLU | 7.63 MiB | 0 MiB |
| Multiply + Add + ReLU | 11.44 MiB | 0 MiB |
| Long Chain | 19.07 MiB | 0 MiB |

These are analytical estimates of allocated intermediate-array
storage, not measurements of peak memory or DRAM traffic.

## Correctness

The existing backend and C++ correctness scripts passed.

Additional checks:

- 60 fusion correctness checks passed
- 6 compiler optimization checks passed
- 36 edge-case checks passed
- 5 pytest wrapper tests passed

## Limitations

- Results were collected on one machine.
- Only two independent benchmark runs were performed.
- Some configurations showed substantial run-to-run variation.
- Measurements include Python/PyTorch invocation and output-allocation
  overhead, not just native kernel execution.
- The backend supports only a limited set of operations and tensor types.
- Memory estimates are analytical, not measured peak memory.
- Generated assembly and vectorization reports were inspected for
  the long-chain workload.

## Reproduction

Install project dependencies and pytest, then run:

```bash
PYTHONPATH=. python3 -m pytest tests/ -q
PYTHONPATH=. python3 benchmarks/benchmark_optimization.py
PYTHONPATH=. python3 benchmarks/plot_optimization.py
PYTHONPATH=. python3 benchmarks/analyze_memory.py
```

The benchmark currently writes its output to the configured CSV
filename in `benchmarks/benchmark_optimization.py`.
