# PyTorch Compiler Backend

A small custom CPU compiler backend for PyTorch inference.

The project explores how compiler optimizations such as operator fusion
and CPU vectorization affect the performance of tensor computations.

## Goals

- Integrate a custom backend with `torch.compile`
- Inspect and process PyTorch FX computation graphs
- Generate C++ kernels for a limited set of tensor operations
- Implement elementwise operator fusion
- Explore CPU SIMD vectorization
- Benchmark against PyTorch eager execution and TorchInductor

## Current Progress

Progress Report 1 focuses on PyTorch integration.

The current implementation:

1. Defines a small PyTorch model.
2. Runs the model normally with PyTorch.
3. Passes the model through `torch.compile`.
4. Uses a custom backend function.
5. Receives and prints the FX graph.
6. Inspects each node in the graph.
7. Checks that the compiled and eager outputs match.

The test model performs:

```text
input
  ↓
multiply by 2
  ↓
add 3
  ↓
ReLU
  ↓
output