"""Run many (config, seed) experiments in parallel, one subprocess per run.

Built for cloud machines (e.g. Runpod CPU pods) that can be stopped at any time:

* every run is started with ``--resume`` and checkpoints every generation by default, so
  re-launching the same sweep continues where it stopped and skips finished runs;
* the number of parallel jobs defaults to the CPUs actually granted to the container
  (cgroup quota), not the host's core count;
* each run logs to ``<run dir>/run.log``; a summary is written to
  ``<out>/sweep_<timestamp>.csv``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .metrics.logger import run_dir


def available_cpus() -> int:
    """CPUs usable by this process: affinity mask, capped by the cgroup CPU quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        n = os.cpu_count() or 1
    quota = _cgroup_quota()
    if quota is not None:
        n = min(n, max(1, math.floor(quota)))
    return max(1, n)


def _cgroup_quota() -> float | None:
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def parse_seeds(text: str) -> list[int]:
    """'1-4,10,12-13' -> [1, 2, 3, 4, 10, 12, 13]."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"invalid or duplicated seeds: {text!r}")
    return seeds


@dataclass
class Task:
    config: Path
    seed: int
    experiment: str
    generations: int
    directory: Path

    def next_generation(self) -> int:
        ckpt = self.directory / "checkpoint.json"
        if not ckpt.exists():
            return 0
        try:
            return int(json.loads(ckpt.read_text())["next_generation"])
        except (OSError, ValueError, KeyError):
            return 0

    def done(self) -> bool:
        return self.next_generation() >= self.generations


def build_tasks(configs: list[str], seeds: list[int], out: str, generations: int | None):
    tasks = []
    for path in configs:
        cfg = load_config(path)
        for seed in seeds:
            cfg.seed = seed
            gens = generations or cfg.genetics.generations
            tasks.append(Task(Path(path), seed, cfg.experiment, gens, run_dir(cfg, out)))
    return tasks


def run_sweep(
    configs: list[str],
    seeds: list[int],
    out: str,
    jobs: int | None = None,
    workers: int = 1,
    generations: int | None = None,
    checkpoint_every: int = 1,
    poll_s: float = 2.0,
) -> list[dict]:
    tasks = build_tasks(configs, seeds, out, generations)
    jobs = jobs or max(1, available_cpus() // workers)
    pending = [t for t in tasks if not t.done()]
    print(
        f"[sweep] {len(tasks)} runs, {len(tasks) - len(pending)} already finished, "
        f"{jobs} parallel jobs x {workers} worker(s), output: {out}",
        flush=True,
    )
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    running: list[tuple[Task, subprocess.Popen, float]] = []
    results: dict[tuple[str, int], dict] = {}
    queue = list(pending)
    try:
        while queue or running:
            while queue and len(running) < jobs:
                t = queue.pop(0)
                t.directory.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-m",
                    "spiking_evo",
                    "run",
                    str(t.config),
                    "--seed",
                    str(t.seed),
                    "--out",
                    out,
                    "--resume",
                    "--workers",
                    str(workers),
                    "--checkpoint-every",
                    str(checkpoint_every),
                ]
                if generations:
                    cmd += ["--generations", str(generations)]
                log = (t.directory / "run.log").open("a")
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                log.close()
                running.append((t, proc, time.perf_counter()))
                print(
                    f"[sweep] start {t.experiment} seed={t.seed} "
                    f"(from generation {t.next_generation()})",
                    flush=True,
                )
            time.sleep(poll_s)
            still = []
            for t, proc, t0 in running:
                code = proc.poll()
                if code is None:
                    still.append((t, proc, t0))
                    continue
                status = "done" if code == 0 and t.done() else f"failed (exit {code})"
                results[(t.experiment, t.seed)] = _summary(t, status, time.perf_counter() - t0)
                print(
                    f"[sweep] {status}: {t.experiment} seed={t.seed} "
                    f"(log: {t.directory / 'run.log'})",
                    flush=True,
                )
            running = still
    except KeyboardInterrupt:
        for _, proc, _ in running:
            proc.terminate()
        raise
    rows = [
        results.get((t.experiment, t.seed)) or _summary(t, "skipped (finished)", 0.0) for t in tasks
    ]
    summary = Path(out) / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] summary: {summary}", flush=True)
    return rows


def _summary(t: Task, status: str, elapsed: float) -> dict:
    last: dict = {}
    gens = t.directory / "generations.csv"
    if gens.exists():
        with gens.open(newline="") as f:
            for last in csv.DictReader(f):  # noqa: B007
                pass
    return {
        "experiment": t.experiment,
        "seed": t.seed,
        "status": status,
        "last_generation": last.get("generation", ""),
        "fitness_mean": last.get("fitness_mean", ""),
        "fitness_max": last.get("fitness_max", ""),
        "elapsed_s": round(elapsed, 1),
        "directory": str(t.directory),
    }
