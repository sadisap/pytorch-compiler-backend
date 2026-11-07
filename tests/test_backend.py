
import torch

from src.backend import make_cpp_backend
from src.model import TinyModel


def main():
    torch.manual_seed(0)

    inputs = [
        torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0]),
        torch.tensor([0.0]),
        torch.randn(10),
        torch.randn(100),
        torch.empty(0),
    ]

    for fuse in (False, True):
        mode = "FUSED" if fuse else "UNFUSED"

        for i, x in enumerate(inputs, start=1):
            torch._dynamo.reset()

            model = TinyModel().eval()

            compiled = torch.compile(
                model,
                backend=make_cpp_backend(fuse=fuse),
                dynamic=False,
                fullgraph=True,
            )

            expected = model(x)
            actual = compiled(x)

            torch.testing.assert_close(
                actual, expected, rtol=1e-5, atol=1e-6
            )

            print(f"{mode} test {i}: PASS")

    print("\nAll backend correctness checks passed.")


if __name__ == "__main__":
    main()
