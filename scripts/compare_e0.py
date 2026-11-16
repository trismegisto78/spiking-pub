"""Stage E0 analysis: do agents evolve to use the others as a cue to find food?

Hypotheses and tests fixed before running (pre-registered):

Two conditions, seeds 1-24 each, 200 generations, groups of 10 agents per world, patchy food
that relocates every 250 steps and is sensed only within 2 cells; the others are sensed
within 6 cells. e0_social: others.sense_gain 1; e0_control: others.sense_gain 0 (same
network inputs, always 0).

Per run:
    energy_late = mean of energy_gained_mean over the last 20 generations (generations.csv)
    blind_cost  = energy gained by the final population (checkpoint.json, 200 genomes, in
                  groups of 10 as in evolution) minus the same population blind to `others`;
                  10 fresh episodes each (generation keys >= 10^6, never used in evolution)

H1 (social information pays): energy_late is higher in e0_social than in e0_control.
    One-sided permutation test on seed-level values (20 000 permutations), alpha 0.05.
H2 (evolved agents rely on the others): blind_cost > 0 in e0_social.
    One-sided sign-flip permutation test on the 24 per-seed values (20 000), alpha 0.05.
Exploratory, no test: blind_cost in e0_control (must be exactly 0: sanity check); per-seed
values; energy_late difference in percent.

    python scripts/compare_e0.py RUNS_DIR [JOBS]
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count() or 1
N_PERM = 20000
LAST = 20
EPISODES = 10
EVAL_KEY_BASE = 1_000_000
CONDITIONS = ("e0_social", "e0_control")


def energy_late(run: Path) -> float:
    rows = list(csv.DictReader((run / "generations.csv").open(newline="")))
    return float(np.mean([float(r["energy_gained_mean"]) for r in rows[-LAST:]]))


def blind_cost(run: str) -> float:
    from spiking_evo.config import config_from_dict
    from spiking_evo.genetics.population import Evolution, load_population

    raw = yaml.safe_load((Path(run) / "config.yaml").read_text(encoding="utf-8"))
    raw.pop("_run", None)
    cfg = config_from_dict(raw)
    genomes, _ = load_population(run)
    out = []
    for blind in ((), ("others",)):
        evo = Evolution(cfg, blind=blind)
        pop = [g.repair(evo.spec) for g in genomes]
        out.append(
            np.mean(
                [
                    evo.evaluate(pop, EVAL_KEY_BASE + k).stats.energy_gained.mean()
                    for k in range(EPISODES)
                ]
            )
        )
    return float(out[0] - out[1])


def perm_greater(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    cnt = sum(
        (p := rng.permutation(pool))[: len(a)].mean() - p[len(a) :].mean() >= obs
        for _ in range(N_PERM)
    )
    return obs, cnt / N_PERM


def signflip_greater(x, rng):
    x = np.asarray(x, float)
    obs = x.mean()
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(x)))
    return obs, float(((signs * np.abs(x)).mean(axis=1) >= obs).mean())


def main() -> None:
    rng = np.random.default_rng(0)
    runs = {
        c: sorted((ROOT / c).glob("*/generations.csv"), key=lambda p: int(p.parent.name))
        for c in CONDITIONS
    }
    late = {c: [energy_late(p.parent) for p in runs[c]] for c in CONDITIONS}
    with ProcessPoolExecutor(JOBS) as pool:
        cost = {c: list(pool.map(blind_cost, [str(p.parent) for p in runs[c]])) for c in CONDITIONS}
    for c in CONDITIONS:
        print(
            f"{c:11s} runs {len(runs[c]):2d}  energy_late {np.mean(late[c]):7.2f}"
            f"  blind_cost {np.mean(cost[c]):+7.2f}"
        )
        for p, e, b in zip(runs[c], late[c], cost[c], strict=True):
            print(f"   {p.parent.name:>3s} energy_late={e:7.2f} blind_cost={b:+7.2f}")
    s, k = CONDITIONS
    print("\n=== Pre-registered tests (one-sided permutation, alpha 0.05)")
    d, p = perm_greater(late[s], late[k], rng)
    print(
        f"H1 energy_late social - control = {d:+.2f} ({100 * d / np.mean(late[k]):+.1f}%),"
        f" p = {p:.4f}"
    )
    d, p = signflip_greater(cost[s], rng)
    print(f"H2 blind_cost in social = {d:+.2f}, p = {p:.4f}")
    print(f"(sanity) blind_cost in control = {np.mean(cost[k]):+.4f} (expected 0)")


if __name__ == "__main__":
    main()
