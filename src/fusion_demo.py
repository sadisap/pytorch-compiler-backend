
import torch

from backend import make_cpp_backend
from model import TinyModel


def main():
    model = TinyModel().eval()

    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0])

    eager = model(x)

    print("Input:", x)
    print("PyTorch:", eager)

    for fuse in (False, True):
        mode = "FUSED" if fuse else "UNFUSED"

        torch._dynamo.reset()

        compiled = torch.compile(
            model,
            backend=make_cpp_backend(fuse=fuse),
            dynamic=False,
        )

        actual = compiled(x)

        print(f"\n{mode} output:", actual)

        torch.testing.assert_close(actual, eager, rtol=1e-5, atol=1e-6)

        print(f"{mode} correctness: PASS")

    print("\nBoth versions passed.")


if __name__ == "__main__":
    main()
