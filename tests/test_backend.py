import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend import inspect_backend
from model import TinyModel


def main():
    model = TinyModel()
    model.eval()

    compiled_model = torch.compile(
        model,
        backend=inspect_backend,
    )

    test_inputs = [
        torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.randn(10),
        torch.randn(100),
    ]

    for i, x in enumerate(test_inputs, start=1):
        eager_output = model(x)
        compiled_output = compiled_model(x)

        assert torch.allclose(eager_output, compiled_output)

        print(f"Test {i}: PASS")

    print("\nAll backend tests passed.")


if __name__ == "__main__":
    main()