
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"


def load(filename):
    with (ROOT / filename).open() as f:
        rows = list(csv.DictReader(f))

    return {
        (r["workload"], int(r["size"]), r["opt_level"], r["mode"]):
        float(r["median_ms"])
        for r in rows
    }


run1 = load("optimization_results.csv")
run2 = load("optimization_results_run2.csv")

assert len(run1) == len(run2) == 96
assert run1.keys() == run2.keys()

print(
    f"{'Workload':22} {'Opt':5} "
    f"{'Run 1':>10} {'Run 2':>10} {'Change':>10}"
)
print("-" * 65)

for workload in sorted({k[0] for k in run1}):
    for opt in ("O0", "O2", "O3"):
        key_u = (workload, 1_000_000, opt, "unfused")
        key_f = (workload, 1_000_000, opt, "fused")

        speedup1 = run1[key_u] / run1[key_f]
        speedup2 = run2[key_u] / run2[key_f]

        change = (speedup2 / speedup1 - 1) * 100

        print(
            f"{workload:22} {opt:5} "
            f"{speedup1:9.2f}x {speedup2:9.2f}x "
            f"{change:+9.1f}%"
        )
