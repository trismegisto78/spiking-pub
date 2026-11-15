"""Plasticity test: fixed vs plastic brains in a static vs a changing world.

python scripts/compare_dynamic.py RUNS_DIR
"""

import sys

import numpy as np

from spiking_evo.analysis import load_runs

ex = load_runs(sys.argv[1] if len(sys.argv) > 1 else "runs")


def row(name, g0, g1):
    e = ex.get(name)
    if e is None or e.generations < g1:
        return f"{name:24s} (missing)"
    f = e.gen["fitness_mean"][:, g0:g1].mean(axis=1)
    alive = e.gen["alive_fraction"][:, g0:g1].mean()
    get = lambda k: np.nanmean(e.stim[k][:, g0:g1]) if k in e.stim else float("nan")  # noqa: E731
    harv = np.nanmean(e.stim["terrain.harvested"][:, g0:g1], axis=1)
    out = (
        f"{name:24s} fitness {f.mean():7.0f} ± {f.std():6.0f}  alive {alive:4.0%}  "
        f"farmers {(harv > 50).sum()}/{len(e.seeds)}  harvests {get('terrain.harvested'):6.1f}  "
        f"hazard steps {get('hazard.steps_inside'):5.1f}  poisoned {get('terrain.poisoned'):5.1f}"
    )
    if "brain.learning_rate" in e.stim:
        out += (
            f"  lr {get('brain.learning_rate'):.4f}  dw {get('brain.weight_change'):.3f}"
            f"  plastic {get('brain.plastic_share'):.2f}"
        )
    return out


names = ["static_fixed_50", "static_plastic_50", "dynamic_fixed_50", "dynamic_plastic_50"]
for g0, g1, label in (
    (0, 10, "first 10 generations"),
    (140, 150, "generations 140-149"),
    (290, 300, "last 10 generations"),
):
    print(f"=== {label}")
    for n in names:
        print(row(n, g0, g1))
print("=== learning rate over generations (plastic runs, mean of seeds)")
for n in ("static_plastic_50", "dynamic_plastic_50"):
    e = ex.get(n)
    if e is not None and "brain.learning_rate" in e.stim:
        lr = e.stim["brain.learning_rate"]
        print(
            n,
            {
                g: round(float(np.nanmean(lr[:, g])), 4)
                for g in (0, 25, 50, 100, 200, e.generations - 1)
            },
        )
