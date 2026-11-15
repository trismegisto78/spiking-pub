"""Command line entry point.

python -m spiking_evo run spiking_evo/experiments/food_only_baseline.yaml
python -m spiking_evo run CONFIG --seed 7 --generations 50 --out runs --resume
python -m spiking_evo evaluate runs/<exp>/<seed>/snapshots/gen_0099.json --config CONFIG
python -m spiking_evo sweep CONFIG [CONFIG ...] --seeds 1-16 --out /workspace/runs
python -m spiking_evo analyze runs --out report
python -m spiking_evo behaviour SNAPSHOT --config CONFIG --episodes 20

The output root defaults to $SPIKING_EVO_OUT, then to metrics.out_dir of the config.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

from .config import load_config
from .genetics.genome import Genome
from .genetics.population import Evolution
from .metrics.logger import run_dir
from .sweep import available_cpus, parse_seeds, run_sweep


def _load(args) -> object:
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    if getattr(args, "generations", None) is not None:
        cfg.genetics.generations = args.generations
    if getattr(args, "checkpoint_every", None) is not None:
        cfg.metrics.checkpoint_every = args.checkpoint_every
    return cfg


def _out(args) -> str | None:
    return args.out or os.environ.get("SPIKING_EVO_OUT") or None


def cmd_run(args) -> int:
    cfg = _load(args)
    evo = Evolution(cfg, workers=args.workers)
    out_root = _out(args)
    out = run_dir(cfg, out_root)
    print(f"[spiking-evo] {cfg.experiment} seed={cfg.seed} -> {out}")
    print(f"[spiking-evo] inputs: {', '.join(evo.input_names)}")

    def progress(row: dict) -> None:
        if not args.quiet:
            print(
                f"gen {row['generation']:4d}  fit mean {row['fitness_mean']:8.3f}  "
                f"max {row['fitness_max']:8.3f}  energy+ {row['energy_gained_mean']:7.2f}  "
                f"rate {row['firing_rate_mean']:6.1f}Hz  syn {row['synapses_mean']:5.1f}  "
                f"hid {row['hidden_mean']:4.1f}  {row['elapsed_s']:.1f}s",
                flush=True,
            )

    evo.run(out_root=out_root, resume=args.resume, progress=progress, init_from=args.init_from)
    return 0


def cmd_evaluate(args) -> int:
    """Re-evaluate a saved genome on fresh episodes (generation keys >= 10^6)."""
    cfg = _load(args)
    with open(args.snapshot, encoding="utf-8") as f:
        data = json.load(f)
    evo = Evolution(cfg)
    genome = Genome.from_dict(data.get("genome", data)).repair(evo.spec)
    results = [evo.evaluate([genome], 1_000_000 + k) for k in range(args.episodes)]
    summary = {
        "episodes": args.episodes,
        "fitness_mean": float(np.mean([r.fitness[0] for r in results])),
        "energy_gained_mean": float(np.mean([r.stats.energy_gained[0] for r in results])),
        "firing_rate_mean_hz": float(np.mean([r.stats.firing_rate_hz[0] for r in results])),
        "visited_cells_mean": float(np.mean([r.stats.visited_cells[0] for r in results])),
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_sweep(args) -> int:
    rows = run_sweep(
        args.configs,
        parse_seeds(args.seeds),
        out=_out(args) or "runs",
        jobs=args.jobs,
        workers=args.workers,
        generations=args.generations,
        checkpoint_every=args.checkpoint_every,
    )
    return 0 if all(not r["status"].startswith("failed") for r in rows) else 1


def cmd_analyze(args) -> int:
    from .analysis import analyze

    res = analyze(args.root, args.out, tail=args.tail)
    print(f"experiments: {', '.join(res['experiments'])}")
    print(f"summary: {res['summary']}")
    for img in res["plots"]:
        print(f"plot: {img}")
    if not res["plots"]:
        print('no plots: install matplotlib (pip install -e ".[analysis]")')
    return 0


def cmd_behaviour(args) -> int:
    from .behaviour import behaviour, format_report

    cfg = _load(args)
    with open(args.snapshot, encoding="utf-8") as f:
        data = json.load(f)
    genome = Genome.from_dict(data.get("genome", data))
    report = behaviour(cfg, genome, episodes=args.episodes)
    text = format_report(report)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="spiking-evo")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an evolutionary experiment")
    r.add_argument("config")
    r.add_argument("--seed", type=int)
    r.add_argument("--generations", type=int)
    r.add_argument("--out", help="root output directory (default: metrics.out_dir)")
    r.add_argument("--resume", action="store_true", help="continue from checkpoint.json")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--checkpoint-every", type=int, help="override metrics.checkpoint_every")
    r.add_argument(
        "--init-from",
        help="start from the final population of another run (its directory or checkpoint)",
    )
    r.add_argument(
        "--workers", type=int, default=1, help="processes evaluating the population (default 1)"
    )
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="re-evaluate a genome snapshot")
    e.add_argument("snapshot")
    e.add_argument("--config", required=True)
    e.add_argument("--seed", type=int)
    e.add_argument("--episodes", type=int, default=10)
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("sweep", help="run several configs x seeds in parallel, resumable")
    s.add_argument("configs", nargs="+")
    s.add_argument("--seeds", required=True, help="e.g. 1-16 or 1,2,7")
    s.add_argument("--out", help="root output directory (default: $SPIKING_EVO_OUT or runs)")
    s.add_argument("--jobs", type=int, help="parallel runs (default: CPUs / workers)")
    s.add_argument("--workers", type=int, default=1, help="processes per run (default 1)")
    s.add_argument("--generations", type=int)
    s.add_argument("--checkpoint-every", type=int, default=1)
    s.set_defaults(func=cmd_sweep)

    a = sub.add_parser("analyze", help="summarise runs across seeds: tables and plots")
    a.add_argument("root", help="runs directory (contains <experiment>/<seed>/)")
    a.add_argument("--out", default="report")
    a.add_argument("--tail", type=int, default=10, help="last generations to average")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("behaviour", help="test which senses an evolved genome relies on")
    b.add_argument("snapshot")
    b.add_argument("--config", required=True)
    b.add_argument("--seed", type=int)
    b.add_argument("--episodes", type=int, default=50)
    b.add_argument("--json", help="also write the report as JSON")
    b.set_defaults(func=cmd_behaviour)

    c = sub.add_parser("cpus", help="print the number of CPUs available to this container")
    c.set_defaults(func=lambda a: print(available_cpus()) or 0)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
