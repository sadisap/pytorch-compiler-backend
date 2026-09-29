import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "src"
    ),
)

from backend import cpp_backend
from model import TinyModel


def run_test(size):
    torch._dynamo.reset()

    model = TinyModel()
    model.eval()

    compiled_model = torch.compile(
        model,
        backend=cpp_backend,
        dynamic=False,
    )

    x = torch.randn(
        size,
        dtype=torch.float32,
    )

    eager_output = model(x)
    backend_output = compiled_model(x)

    assert torch.allclose(
        eager_output,
        backend_output,
    )

    print(
        f"Size {size}: PASS"
    )


def main():
    sizes = [
        1,
        5,
        10,
        100,
        1000,
    ]

    for size in sizes:
        run_test(size)

    print(
        "\nAll C++ backend correctness tests passed."
    )


if __name__ == "__main__":
    main()