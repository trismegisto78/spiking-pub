"""Command line entry point.

python -m spiking_evo run spiking_evo/experiments/food_only_baseline.yaml
python -m spiking_evo run CONFIG --seed 7 --generations 50 --out runs --resume
python -m spiking_evo evaluate runs/<exp>/<seed>/snapshots/gen_0099.json --config CONFIG
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np

from .config import load_config
from .genetics.genome import Genome
from .genetics.population import Evolution
from .metrics.logger import run_dir


def _load(args) -> object:
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    if getattr(args, "generations", None) is not None:
        cfg.genetics.generations = args.generations
    return cfg


def cmd_run(args) -> int:
    cfg = _load(args)
    evo = Evolution(cfg)
    out = run_dir(cfg, args.out)
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

    evo.run(out_root=args.out, resume=args.resume, progress=progress)
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
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="re-evaluate a genome snapshot")
    e.add_argument("snapshot")
    e.add_argument("--config", required=True)
    e.add_argument("--seed", type=int)
    e.add_argument("--episodes", type=int, default=10)
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
