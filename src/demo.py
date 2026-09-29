import torch

from backend import cpp_backend
from model import TinyModel


def main():
    model = TinyModel()
    model.eval()

    x = torch.tensor(
        [-3.0, -1.0, 0.0, 1.0, 2.0],
        dtype=torch.float32,
    )

    print("Input:")
    print(x)

    print("\n=== Eager PyTorch ===")
    eager_output = model(x)
    print(eager_output)

    print("\n=== Custom C++ Backend ===")

    compiled_model = torch.compile(
        model,
        backend=cpp_backend,
        dynamic=False,
    )

    compiled_output = compiled_model(x)

    print("\nC++ backend output:")
    print(compiled_output)

    print("\n=== Correctness Check ===")

    matches = torch.allclose(
        eager_output,
        compiled_output,
    )

    print("Outputs match:", matches)

    if not matches:
        raise RuntimeError(
            "C++ backend output does not match PyTorch."
        )


if __name__ == "__main__":
    main()