"""Stage E1 analysis: does a signal that carries information evolve, and does kinship matter?

Hypotheses and tests fixed before running (pre-registered, 16/11/2026):

2x2 design, seeds 1-24 per condition, 300 generations, the E0 world (groups of 10 agents,
patchy relocating food sensed within 2 cells, the others sensed within 6) plus a `signal`
channel (one action output, emit cost 0.01, heard by the group at the next step within 6
cells). 400 agents per generation in every condition.
    e1_clones       groups of 10 clones of one genome, fitness = group mean, signal heard
    e1_clones_mute  same, signal.sense_gain 0 (nobody hears anything)
    e1_mixed        400 unrelated genomes, groups re-drawn every episode, individual
                    selection, signal heard
    e1_mixed_mute   same, mute

Per run, on the final population (checkpoint.json) evaluated as in evolution on 10 fresh
episodes (generation keys >= 10^6, never used during evolution), pooling all agents:
    informativeness = P(emit | fed) - P(emit | unfed), where fed = energy gained from the
                      stimuli in one of the last 10 steps (signal.* episode statistics)
    emission_rate   = emissions / living agent-steps
    blind_cost      = energy gained (normal) - energy gained when deaf to `signal`
                      (same population, same episodes; emitting and its cost unchanged)
    energy_late     = mean of energy_gained_mean over the last 20 generations (generations.csv)

H1 (an informative signal evolves among kin): informativeness higher in e1_clones than in
    e1_clones_mute. One-sided permutation test on seed-level values.
H2 (kin use the signal): blind_cost > 0 in e1_clones. One-sided sign-flip test on the
    per-seed values.
H3 (kinship matters): informativeness higher in e1_clones than in e1_mixed. One-sided
    permutation test.
All tests: 20 000 permutations, alpha 0.05, no correction (three distinct questions).
Exploratory, no test: energy_late, emission_rate and informativeness in all four
conditions; e1_mixed vs e1_mixed_mute; blind_cost in e1_mixed.
Sanity check: blind_cost is exactly 0 in both mute conditions.

    python scripts/compare_e1.py RUNS_DIR [JOBS]
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
CONDITIONS = ("e1_clones", "e1_clones_mute", "e1_mixed", "e1_mixed_mute")
METRICS = ("energy_late", "informativeness", "emission_rate", "blind_cost")


def energy_late(run: Path) -> float:
    rows = list(csv.DictReader((run / "generations.csv").open(newline="")))
    return float(np.mean([float(r["energy_gained_mean"]) for r in rows[-LAST:]]))


def final_population_metrics(run: str) -> dict[str, float]:
    from spiking_evo.config import config_from_dict
    from spiking_evo.genetics.population import Evolution, load_population

    raw = yaml.safe_load((Path(run) / "config.yaml").read_text(encoding="utf-8"))
    raw.pop("_run", None)
    cfg = config_from_dict(raw)
    genomes, _ = load_population(run)
    energy = {}
    counts = dict.fromkeys(("emitted", "steps_fed", "emitted_fed", "steps_unfed",
                            "emitted_unfed"), 0.0)  # fmt: skip
    for blind in ((), ("signal",)):
        evo = Evolution(cfg, blind=blind)
        pop = [g.repair(evo.spec) for g in genomes]
        gained = []
        for k in range(EPISODES):
            ev = evo.evaluate(pop, EVAL_KEY_BASE + k)
            gained.append(ev.stats.energy_gained.mean())
            if not blind:
                for c in counts:
                    counts[c] += float(ev.stimulus_stats[f"signal.{c}"].sum())
        energy[blind] = float(np.mean(gained))
    p_fed = counts["emitted_fed"] / max(counts["steps_fed"], 1.0)
    p_unfed = counts["emitted_unfed"] / max(counts["steps_unfed"], 1.0)
    steps = counts["steps_fed"] + counts["steps_unfed"]
    return {
        "informativeness": p_fed - p_unfed,
        "emission_rate": counts["emitted"] / max(steps, 1.0),
        "blind_cost": energy[()] - energy[("signal",)],
    }


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
    paths = [str(p.parent) for c in CONDITIONS for p in runs[c]]
    with ProcessPoolExecutor(JOBS) as pool:
        final = dict(zip(paths, pool.map(final_population_metrics, paths), strict=True))
    data = {
        c: [
            {"seed": p.parent.name, "energy_late": energy_late(p.parent), **final[str(p.parent)]}
            for p in runs[c]
        ]  # fmt: skip
        for c in CONDITIONS
    }
    for c in CONDITIONS:
        means = "  ".join(
            f"{m} {np.mean([r[m] for r in data[c]]) if data[c] else np.nan:+.4f}" for m in METRICS
        )
        print(f"{c:15s} runs {len(data[c]):2d}  {means}")
        for r in data[c]:
            vals = " ".join(f"{m}={r[m]:+.4f}" for m in METRICS)
            print(f"   {r['seed']:>3s} {vals}")

    def v(c, m):
        return [r[m] for r in data[c]]

    print("\n=== Pre-registered tests (one-sided permutation, alpha 0.05)")
    d, p = perm_greater(v("e1_clones", "informativeness"), v("e1_clones_mute", "informativeness"),
                        rng)  # fmt: skip
    print(f"H1 informativeness clones - clones_mute = {d:+.4f}, p = {p:.4f}")
    d, p = signflip_greater(v("e1_clones", "blind_cost"), rng)
    print(f"H2 blind_cost in clones = {d:+.3f}, p = {p:.4f}")
    d, p = perm_greater(v("e1_clones", "informativeness"), v("e1_mixed", "informativeness"), rng)
    print(f"H3 informativeness clones - mixed = {d:+.4f}, p = {p:.4f}")
    print("(sanity) blind_cost in the mute conditions: "
          f"clones_mute {np.mean(v('e1_clones_mute', 'blind_cost')):+.4f}, "
          f"mixed_mute {np.mean(v('e1_mixed_mute', 'blind_cost')):+.4f} (expected 0)")  # fmt: skip


if __name__ == "__main__":
    main()
