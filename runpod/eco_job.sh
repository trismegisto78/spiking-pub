#!/usr/bin/env bash
# Runpod job: ecosystem runs for CONFIGS x SEEDS in parallel (one process per run), then
# COMPARE (a script taking the runs directory). Results served on port 8000 at the end.
set -x
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/code
pip install -q -e ".[analysis]"
OUT=/workspace/runs
mkdir -p $OUT
JOBS=$(python -m spiking_evo cpus)
SEED_LIST=$(python -c "from spiking_evo.sweep import parse_seeds; print(*parse_seeds('${SEEDS:-1-24}'))")
for c in $CONFIGS; do for s in $SEED_LIST; do echo "$c $s"; done; done |
  xargs -P "$JOBS" -L 1 bash -c 'python -m spiking_evo ecosystem "$0" --seed "$1" --out '"$OUT"' --quiet > /dev/null 2>&1 && echo "done $0 $1" || echo "FAILED $0 $1"'
echo "=== RESULTS"
[ -n "${COMPARE:-}" ] && python $COMPARE $OUT
tar czf /workspace/results.tgz --exclude='snapshots' -C /workspace runs
echo "=== JOB DONE"
cd /workspace && python -m http.server 8000
