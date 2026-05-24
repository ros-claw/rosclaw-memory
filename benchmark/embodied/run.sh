#!/bin/bash
set -e

cd "$(dirname "$0")"

for scale in 10000 100000 1000000; do
    echo "=== Scale: $scale ==="
    python3 bench_embodied.py --scale $scale --output results_${scale}.json
done

echo "=== All benchmarks complete ==="
ls -la results_*.json
