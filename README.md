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

## Status

In development.

Current milestone: receive and inspect PyTorch FX graphs through a custom
`torch.compile` backend.