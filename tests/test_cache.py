
import contextlib
import io
import torch

from src.backend import make_cpp_backend


class CacheModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2 + 3)


def compile_once():
    torch._dynamo.reset()

    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        model = torch.compile(
            CacheModel(),
            backend=make_cpp_backend(fuse=True, opt_level="O3"),
            fullgraph=True,
        )
        x = torch.randn(1000)
        torch.testing.assert_close(model(x), CacheModel()(x))

    return output.getvalue()


if __name__ == "__main__":
    first = compile_once()
    second = compile_once()

    print("First compilation:")
    print("CACHE HIT" if "CACHE HIT" in first else "CACHE MISS")

    print("Second compilation:")
    print("CACHE HIT" if "CACHE HIT" in second else "CACHE MISS")

    assert "CACHE HIT" in second
    print("Cache reuse verified!")
