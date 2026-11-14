#!/usr/bin/env bash
# Pod entrypoint: SSH access, then (optionally) a resumable sweep, then stay alive.
#
# Environment (set in the pod template):
#   SWEEP_CONFIGS  space-separated config paths, e.g.
#                  "/opt/spiking-evo/spiking_evo/experiments/food_only_baseline.yaml"
#                  (empty = do not start anything, just keep the pod up for SSH)
#   SWEEP_SEEDS    seeds, e.g. "1-16" (default 1-8)
#   SWEEP_ARGS     extra arguments for `spiking-evo sweep`, e.g. "--generations 300"
#   SPIKING_EVO_OUT  output root (default /workspace/runs: put a network volume there)
#
# If the pod is stopped or interrupted, starting it again re-runs this script and the
# sweep continues from the last checkpoint of every run.
set -uo pipefail

mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null 2>&1
service ssh start >/dev/null 2>&1 || /usr/sbin/sshd || echo "[start] sshd not started"

OUT="${SPIKING_EVO_OUT:-/workspace/runs}"
mkdir -p "$OUT"
echo "[start] CPUs available: $(python -m spiking_evo cpus); output: $OUT"

if [[ -n "${SWEEP_CONFIGS:-}" ]]; then
    # shellcheck disable=SC2086  # word splitting of the lists is intended
    python -m spiking_evo sweep $SWEEP_CONFIGS --seeds "${SWEEP_SEEDS:-1-8}" \
        --out "$OUT" ${SWEEP_ARGS:-} 2>&1 | tee -a "$OUT/sweep.log"
    echo "[start] sweep finished; stop the pod to stop billing."
fi

# A pod whose main process exits gets restarted: keep it alive instead.
exec sleep infinity
