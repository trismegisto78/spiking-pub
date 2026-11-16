"""Run ecosystem worlds (configs x seeds) in parallel on this machine, with a live status table.

Local equivalent of runpod/eco_job.sh: one process per world, at most JOBS at a time. Each
world writes its progress to OUT/<experiment>/<seed>/run.log; every 30 s the table shows the
last line of every world. Worlds already finished (ecosystem.csv reaching the last tick) are
skipped, so the command can be re-launched after an interruption.

    python scripts/eco_local.py --seeds 1-8 --out runs \\
        spiking_evo/experiments/eco_d6_care.yaml spiking_evo/experiments/eco_d6_control.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from spiking_evo.sweep import available_cpus, parse_seeds


def finished(run_dir: Path, ticks: int) -> bool:
    path = run_dir / "ecosystem.csv"
    if not path.exists():
        return False
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return bool(rows) and int(float(rows[-1]["tick"])) >= ticks


def last_line(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1] if lines else ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--seeds", required=True, help="e.g. 1-8")
    p.add_argument("--out", default="runs")
    p.add_argument("--jobs", type=int, default=available_cpus())
    p.add_argument("--ticks", type=int, help="override ecosystem.ticks")
    p.add_argument("--every", type=float, default=30.0, help="seconds between status tables")
    args = p.parse_args()

    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    todo = []
    for cfg_path in args.configs:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        ticks = args.ticks or cfg["ecosystem"]["ticks"]
        for seed in parse_seeds(args.seeds):
            run_dir = Path(args.out) / cfg["experiment"] / str(seed)
            if finished(run_dir, ticks):
                print(f"skip (done) {cfg['experiment']} {seed}")
                continue
            todo.append((cfg_path, cfg["experiment"], seed, run_dir, ticks))

    running: dict[tuple[str, int], tuple[subprocess.Popen, Path, object]] = {}
    status: dict[tuple[str, int], str] = {}
    start = time.time()
    next_table = 0.0
    while todo or running:
        while todo and len(running) < args.jobs:
            cfg_path, exp, seed, run_dir, ticks = todo.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            log = run_dir / "run.log"
            fh = log.open("w", encoding="utf-8")
            cmd = [
                sys.executable,
                "-m",
                "spiking_evo",
                "ecosystem",
                cfg_path,
                "--seed",
                str(seed),
                "--out",
                args.out,
            ]
            if args.ticks:
                cmd += ["--ticks", str(ticks)]
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[(exp, seed)] = (proc, log, fh)
            status[(exp, seed)] = "started"
        for key, (proc, log, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                status[key] = (
                    "DONE  " if proc.returncode == 0 else f"FAILED (exit {proc.returncode}) "
                ) + last_line(log)
                del running[key]
        now = time.time()
        if now >= next_table or not (todo or running):
            elapsed = (now - start) / 60
            print(
                f"\n=== {time.strftime('%H:%M:%S')}  {elapsed:.1f} min  "
                f"running {len(running)}  queued {len(todo)}",
                flush=True,
            )
            for key in sorted(status):
                line = status[key]
                if key in running:
                    line = last_line(running[key][1]) or "starting"
                print(f"{key[0]:>16s} {key[1]:>3d} | {line}", flush=True)
            next_table = now + args.every
        time.sleep(1.0)
    failed = [k for k, s in status.items() if s.startswith("FAILED")]
    print(f"\nall done in {(time.time() - start) / 60:.1f} min, failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
