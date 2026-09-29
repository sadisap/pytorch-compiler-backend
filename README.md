# PyTorch Compiler Backend

A small custom CPU compiler backend for PyTorch inference.

The project explores how compiler optimizations such as operator fusion
and CPU vectorization affect the performance of tensor computations.

## Goals

- Integrate a custom backend with `torch.compile`
- Inspect and process PyTorch FX computation graphs
- Generate C++ code for a limited set of tensor operations
- Implement elementwise operator fusion
- Explore CPU SIMD vectorization
- Benchmark against PyTorch eager execution and TorchInductor

## Current Progress

The project now has a basic end-to-end C++ backend for a small set of
PyTorch operations.

The current pipeline is:

1. Define a small PyTorch model.
2. Pass the model through `torch.compile`.
3. Receive the FX graph in the custom backend.
4. Read the supported operations from the graph.
5. Generate C++ code for those operations.
6. Compile the generated C++ into a shared library.
7. Load and execute the compiled C++ from Python.
8. Return the result as a PyTorch tensor.
9. Compare the result with normal PyTorch execution.

The current test model performs:

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