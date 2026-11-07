
import torch

from src.backend import make_cpp_backend
from src.model import TinyModel


def main():
    torch.manual_seed(42)

    for opt_level in ("O0", "O2", "O3"):
        for fuse in (False, True):
            torch._dynamo.reset()

            model = TinyModel().eval()
            x = torch.randn(1000)

            compiled = torch.compile(
                model,
                backend=make_cpp_backend(
                    fuse=fuse,
                    opt_level=opt_level,
                ),
                fullgraph=True,
            )

            expected = model(x)
            actual = compiled(x)

            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-6,
            )

            mode = "FUSED" if fuse else "UNFUSED"
            print(f"{opt_level} | {mode}: PASS")

    print("\nAll 6 optimization checks passed.")


if __name__ == "__main__":
    main()
