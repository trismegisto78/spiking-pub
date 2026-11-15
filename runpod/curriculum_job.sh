#!/usr/bin/env bash
# Runpod job: evolutionary path with and without plasticity vs populations from scratch.
# Clone https://github.com/trismegisto78/spiking-pub into /workspace/code, then run this.
# Results: /workspace/runs, summary in the pod log, files served on port 8000 at the end.
set -x
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/code
pip install -q -e ".[analysis]"
E=spiking_evo/experiments
OUT=/workspace/runs
SEEDS=${SEEDS:-1-8}
python -m spiking_evo cpus
python -m spiking_evo sweep $E/curriculum_50.yaml $E/curriculum_50_plastic.yaml \
    $E/terrain_50_spicy.yaml --seeds "$SEEDS" --out $OUT --checkpoint-every 10
for s in $(python -c "from spiking_evo.sweep import parse_seeds; print(*parse_seeds('$SEEDS'))"); do
  python -m spiking_evo run $E/shock_50.yaml --seed $s --out $OUT --quiet \
      --init-from $OUT/terrain_50_spicy/$s > $OUT/shock_$s.log 2>&1 &
done
wait
echo "=== RESULTS"
python scripts/compare_curriculum.py $OUT
python -m spiking_evo analyze $OUT --out $OUT/report
cat $OUT/report/summary.md
tar czf /workspace/results.tgz --exclude=checkpoint.json -C /workspace runs
echo "=== JOB DONE"
cd /workspace && python -m http.server 8000
