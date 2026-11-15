"""Compare the evolutionary path (curriculum_50) with and without plasticity, and with
populations evolved from scratch in the final environment (terrain_50_spicy + shock_50).

    python scripts/compare_curriculum.py RUNS_DIR
"""

import sys

import numpy as np

from spiking_evo.analysis import load_runs

ex = load_runs(sys.argv[1] if len(sys.argv) > 1 else "runs")


def block(name, g0, g1):
    e = ex.get(name)
    if e is None or e.generations < g1:
        return None
    f = e.gen["fitness_mean"][:, g0:g1].mean(axis=1)
    alive = e.gen["alive_fraction"][:, g0:g1].mean(axis=1)
    harv = np.nanmean(
        e.stim.get("terrain.harvested", np.zeros_like(e.gen["fitness_mean"]))[:, g0:g1], axis=1
    )
    haz = np.nanmean(
        e.stim.get("hazard.steps_inside", np.zeros_like(e.gen["fitness_mean"]))[:, g0:g1], axis=1
    )
    out = dict(
        seeds=len(e.seeds),
        fit=f.mean(),
        fit_sd=f.std(),
        alive=alive.mean(),
        farmers=int((harv > 50).sum()),
        hazard=haz.mean(),
    )
    for k in ("brain.learning_rate", "brain.weight_change", "brain.plastic_share"):
        if k in e.stim:
            out[k] = float(np.nanmean(e.stim[k][:, g0:g1]))
    return out


def show(label, res):
    if res is None:
        print(f"{label:58s} (missing)")
        return
    extra = "".join(f" {k.split('.')[1]} {v:.3f}" for k, v in res.items() if k.startswith("brain"))
    print(
        f"{label:58s} fitness {res['fit']:7.0f} ± {res['fit_sd']:5.0f}  alive {res['alive']:4.0%}  "
        f"farmers {res['farmers']}/{res['seeds']}  hazard steps {res['hazard']:5.1f}{extra}"
    )


print("=== Evolutionary path, phase by phase (last 10 generations of each phase)")
for g0, g1, phase in (
    (90, 100, "food only"),
    (190, 200, "+ hazards"),
    (290, 300, "+ farming"),
    (440, 450, "spicy seasons"),
    (490, 500, "shock"),
):
    for name in ("curriculum_50", "curriculum_50_plastic"):
        show(f"{phase:14s} {name}", block(name, g0, g1))
print("\n=== Arrival in the spicy environment (first 10 generations)")
show("scratch terrain_50_spicy gens 0-9", block("terrain_50_spicy", 0, 10))
for name in ("curriculum_50", "curriculum_50_plastic"):
    show(f"{name} gens 300-309", block(name, 300, 310))
print("\n=== After 150 generations in the spicy environment")
show("scratch terrain_50_spicy gens 140-149", block("terrain_50_spicy", 140, 150))
for name in ("curriculum_50", "curriculum_50_plastic"):
    show(f"{name} gens 440-449", block(name, 440, 450))
show("scratch terrain_50_spicy gens 290-299", block("terrain_50_spicy", 290, 300))
print("\n=== Shock (harsh winter, half the wild food)")
show("scratch pop shock gens 0-4", block("shock_from_spicy", 0, 5))
for name in ("curriculum_50", "curriculum_50_plastic"):
    show(f"{name} shock gens 450-454", block(name, 450, 455))
show("scratch pop shock gens 40-49", block("shock_from_spicy", 40, 50))
for name in ("curriculum_50", "curriculum_50_plastic"):
    show(f"{name} shock gens 490-499", block(name, 490, 500))
