
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "test_backend.py",
    "test_cpp_backend.py",
    "test_fusion.py",
    "test_optimization.py",
    "test_edge_cases.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"{script} failed:\n"
        f"{result.stdout[-3000:]}\n"
        f"{result.stderr[-3000:]}"
    )

    if script == "test_edge_cases.py":
        assert "36 passed, 0 failed" in result.stdout

    if script == "test_fusion.py":
        assert "All 60 fusion correctness checks passed." in result.stdout

    if script == "test_optimization.py":
        assert "All 6 optimization checks passed." in result.stdout
