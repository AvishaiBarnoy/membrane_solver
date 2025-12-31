import json
import os
import subprocess
import sys

BASELINE_FILE = "benchmarks/inputs/results.json"


def run_check():
    if not os.path.exists(BASELINE_FILE):
        print(f"No baseline file found at {BASELINE_FILE}. Skipping performance check.")
        return

    with open(BASELINE_FILE, "r") as f:
        baseline = json.load(f)

    # Run the existing benchmark suite
    print("Running benchmark suite...")
    subprocess.run([sys.executable, "tools/suite.py"], check=True)

    # Suite updates results.json
    with open(BASELINE_FILE, "r") as f:
        current = json.load(f)

    regressions = []
    for key, data in baseline.items():
        if key not in current:
            continue

        b_time = data.get("time", 0)
        c_time = current[key].get("time", 0)

        if b_time == 0:
            continue

        ratio = c_time / b_time
        print(
            f"Benchmark '{key}': {b_time:.4f}s -> {c_time:.4f}s (ratio: {ratio:.2f}x)"
        )

        if ratio > 1.25:  # Allow 20% noise
            regressions.append(f"{key}: {ratio:.2f}x slowdown")

    if regressions:
        print("\nPERFORMANCE REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        sys.exit(1)
    else:
        print("\nPerformance check passed.")


if __name__ == "__main__":
    run_check()
