#!/usr/bin/env bash
# Generic Runpod job: sweep CONFIGS (space separated, relative to the repo) over SEEDS,
# then run COMPARE (a script taking the runs directory), analyze, and serve the results.
set -x
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/code
pip install -q -e ".[analysis]"
OUT=/workspace/runs
python -m spiking_evo cpus
python -m spiking_evo sweep $CONFIGS --seeds "${SEEDS:-1-8}" --out $OUT --checkpoint-every 10
echo "=== RESULTS"
[ -n "${COMPARE:-}" ] && python $COMPARE $OUT
python -m spiking_evo analyze $OUT --out $OUT/report
tar czf /workspace/results.tgz --exclude=checkpoint.json -C /workspace runs
echo "=== JOB DONE"
cd /workspace && python -m http.server 8000
