"""Stage C: brains that are born small and grow, in the changing world.

    python scripts/compare_development.py RUNS_DIR [PREVIOUS_RUNS_DIR]

PREVIOUS_RUNS_DIR may hold dynamic_fixed_50 / dynamic_plastic_50 from the earlier test.
"""

import sys

import numpy as np

from spiking_evo.analysis import load_runs

ex = load_runs(sys.argv[1] if len(sys.argv) > 1 else "runs")
if len(sys.argv) > 2:
    ex.update({k: v for k, v in load_runs(sys.argv[2]).items() if k not in ex})

NAMES = [
    "dynamic_fixed_50",
    "dynamic_plastic_50",
    "dynamic_awake_cost_50",
    "dynamic_develop_50",
    "dynamic_develop_plastic_new_50",
    "dynamic_develop_plastic_all_50",
]


def row(name, g0, g1):
    e = ex.get(name)
    if e is None or e.generations < g1:
        return f"{name:32s} (missing)"
    f = e.gen["fitness_mean"][:, g0:g1].mean(axis=1)
    alive = e.gen["alive_fraction"][:, g0:g1].mean()
    get = lambda k: np.nanmean(e.stim[k][:, g0:g1]) if k in e.stim else float("nan")  # noqa: E731
    harv = np.nanmean(e.stim["terrain.harvested"][:, g0:g1], axis=1)
    out = (
        f"{name:32s} fitness {f.mean():7.0f} ± {f.std():6.0f}  alive {alive:4.0%}  "
        f"farmers {(harv > 50).sum()}/{len(e.seeds)}  hazard {get('hazard.steps_inside'):5.1f}"
    )
    if "brain.woken" in e.stim:
        out += (
            f"  dormant@birth {get('brain.dormant_at_birth'):4.1f}  woken {get('brain.woken'):4.1f}"
            f"  awake@end {get('brain.awake_at_end'):4.1f}"
            f"  difficulty {get('brain.difficulty'):6.1f}"
        )
    if "brain.learning_rate" in e.stim:
        out += f"  lr {get('brain.learning_rate'):.4f}"
    return out


for g0, g1, label in (
    (0, 10, "first 10 generations"),
    (140, 150, "generations 140-149"),
    (290, 300, "last 10 generations"),
):
    print(f"=== {label}")
    for n in NAMES:
        print(row(n, g0, g1))
