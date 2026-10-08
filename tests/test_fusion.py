
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend import make_cpp_backend
from model import TinyModel


class AddModel(nn.Module):
    def forward(self, x):
        return x + 3


class MultiplyAddModel(nn.Module):
    def forward(self, x):
        return x * 2 + 3


class AddReluModel(nn.Module):
    def forward(self, x):
        return torch.relu(x + 3)


class LongChainModel(nn.Module):
    def forward(self, x):
        x = x * 2
        x = x + 3
        x = torch.relu(x)
        x = x * 0.5
        return x + 1


MODELS = [
    TinyModel,
    AddModel,
    MultiplyAddModel,
    AddReluModel,
    LongChainModel,
]


def check(model_class, x, fuse):
    torch._dynamo.reset()

    model = model_class().eval()

    compiled = torch.compile(
        model,
        backend=make_cpp_backend(fuse=fuse),
        dynamic=False,
    )

    expected = model(x)
    actual = compiled(x)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def main():
    torch.manual_seed(0)

    inputs = [
        torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0]),
        torch.randn(1),
        torch.randn(100),
        torch.randn(1000),
        torch.randn(8, 16),
        torch.randn(8, 16).t(),
    ]

    count = 0

    for model_class in MODELS:
        for x in inputs:
            for fuse in (False, True):
                check(model_class, x, fuse)
                count += 1

        print(f"{model_class.__name__}: PASS")

    print(f"\nAll {count} fusion correctness checks passed.")


if __name__ == "__main__":
    main()
