
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OPERATIONS = {
    "multiply_add": 2,
    "add_relu": 2,
    "multiply_add_relu": 3,
    "long_chain": 5,
}

SIZES = [1_000, 10_000, 100_000, 1_000_000]

rows = []

for workload, operations in OPERATIONS.items():
    for n in SIZES:
        tensor_bytes = n * 4

        # Current unfused codegen creates one temporary
        # std::vector<float> per operation.
        unfused_temp_bytes = operations * tensor_bytes

        # Fused codegen creates no intermediate arrays.
        fused_temp_bytes = 0

        rows.append({
            "workload": workload,
            "size": n,
            "operations": operations,
            "unfused_temp_mib": round(
                unfused_temp_bytes / (1024 ** 2), 3
            ),
            "fused_temp_mib": fused_temp_bytes,
        })

output = ROOT / "results" / "memory_estimates.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} memory estimates to {output}")

print("\nAt 1 million elements:")
for row in rows:
    if row["size"] == 1_000_000:
        print(
            f"{row['workload']:20} "
            f"unfused={row['unfused_temp_mib']:6.2f} MiB "
            f"fused={row['fused_temp_mib']} MiB"
        )
