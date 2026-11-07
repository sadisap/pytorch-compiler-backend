
import concurrent.futures
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER = """
import torch
from src.backend import make_cpp_backend

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2 + 3)

torch._dynamo.reset()
model = Model()
compiled = torch.compile(
    model,
    backend=make_cpp_backend(fuse=True, opt_level="O3"),
    fullgraph=True,
)
x = torch.randn(1000)
torch.testing.assert_close(compiled(x), model(x))
print("WORKER PASS")
"""


def run_worker(_):
    return subprocess.run(
        [sys.executable, "-c", WORKER],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_concurrent_compilation():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_worker, range(4)))

    for i, result in enumerate(results):
        assert result.returncode == 0, (
            f"Worker {i} failed:\n{result.stderr}"
        )
        assert "WORKER PASS" in result.stdout

    print("4 concurrent workers passed!")


if __name__ == "__main__":
    test_concurrent_compilation()
