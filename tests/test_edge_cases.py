
import torch

from src.backend import make_cpp_backend


class ReluModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class AddReluModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 3)


def main():
    inputs = {
        "special_values": torch.tensor([
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.0,
            0.0,
            -3.0,
            2.0,
        ], dtype=torch.float32),
        "empty": torch.empty(0, dtype=torch.float32),
        "noncontiguous": torch.randn(4, 6).T,
    }

    failures = []

    for model_class in (ReluModel, AddReluModel):
        for opt_level in ("O0", "O2", "O3"):
            for fuse in (False, True):
                for name, x in inputs.items():
                    torch._dynamo.reset()

                    model = model_class().eval()

                    try:
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
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )

                        print(
                            f"PASS: {model_class.__name__} "
                            f"{opt_level} fuse={fuse} {name}"
                        )

                    except Exception as exc:
                        label = (
                            f"{model_class.__name__} "
                            f"{opt_level} fuse={fuse} {name}"
                        )
                        failures.append(label)
                        print(f"FAIL: {label}")
                        print(f"  {type(exc).__name__}: {exc}")

    print(
        f"\nCompleted 36 checks: "
        f"{36 - len(failures)} passed, "
        f"{len(failures)} failed."
    )

    if failures:
        print("\nFailing cases:")
        for failure in failures:
            print("-", failure)


if __name__ == "__main__":
    main()
