#!/usr/bin/env bash
# Install spiking-evo on a pod started from a stock image (no custom image needed).
# Run from the repository checkout, e.g. after cloning into /workspace:
#   bash spiking-evo/runpod/setup_on_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install --no-cache-dir -e .
python3 -m spiking_evo cpus >/dev/null
cat <<EOF
spiking-evo installed. CPUs available to this pod: $(python3 -m spiking_evo cpus)
Example (resumable, results on the volume):
  export SPIKING_EVO_OUT=/workspace/runs && mkdir -p /workspace/runs
  nohup python3 -m spiking_evo sweep spiking_evo/experiments/food_only_baseline.yaml \\
      --seeds 1-16 > /workspace/runs/sweep.log 2>&1 &
EOF
