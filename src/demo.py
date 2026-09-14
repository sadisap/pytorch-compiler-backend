import torch

from backend import inspect_backend
from model import TinyModel


def main():
    model = TinyModel()
    model.eval()

    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0])

    print("Input:")
    print(x)

    print("\n=== Eager PyTorch ===")
    eager_output = model(x)
    print(eager_output)

    print("\n=== torch.compile with custom backend ===")
    compiled_model = torch.compile(
        model,
        backend=inspect_backend,
    )

    compiled_output = compiled_model(x)

    print("Compiled output:")
    print(compiled_output)

    print("\n=== Correctness Check ===")
    print("Outputs match:", torch.allclose(eager_output, compiled_output))


if __name__ == "__main__":
    main()