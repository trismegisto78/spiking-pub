"""Diagnose how juveniles and adults die in an ecosystem run: energy budget per life.

    python scripts/diagnose_juveniles.py SEED care|control TICKS [FEED_AMOUNT]

Runs spiking_evo/experiments/eco_d5_<cond>.yaml for TICKS steps. For every agent that dies it
records, over its whole life: age, energy at birth, food eaten, harvest energy, hazard and
poison damage, energy invested in brain growth, energy received from its parent.
FEED_AMOUNT overrides the kin feed amount (only where feeding is enabled).
"""

import sys

import numpy as np

from spiking_evo.config import load_config
from spiking_evo.ecosystem import Ecosystem

seed, cond, ticks = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
cfg = load_config(f"spiking_evo/experiments/eco_d5_{cond}.yaml")
cfg.seed = seed
if len(sys.argv) > 4:
    for s in cfg.stimuli:
        if s["type"] == "kin" and s.get("feed_amount", 2.0) > 0:
            s["feed_amount"] = float(sys.argv[4])

sim = Ecosystem(cfg)
agents = sim.world.agents
stim = {s.name: s for s in sim.world.stimuli}
invest = np.zeros(len(agents))
_invest_in_growth = sim.controller.invest_in_growth


def _hook(amount):
    invest[:] += amount
    return _invest_in_growth(amount)


sim.controller.invest_in_growth = _hook


def snapshot() -> dict[str, np.ndarray]:
    terrain = stim["terrain"]
    raw = {
        "eaten": stim["food"].eaten,
        "hazard": stim["hazard"].damage_taken,
        "poison": terrain.poison_steps * terrain.poison_damage,
        "harvest": terrain.harvest_energy_total,
        "received": stim["kin"].energy_received,
        "invest": invest,
    }
    return {k: np.array(v, float).copy() for k, v in raw.items()}


base = snapshot()
birth_energy = agents.energy.copy()
records: list[dict] = []
births, pops = 0, []
for _ in range(ticks):
    alive0, stage0 = agents.alive.copy(), agents.stage.copy()
    sim.step()
    cur = snapshot()
    for i in np.flatnonzero(alive0 & ~agents.alive):
        life = {k: cur[k][i] - base[k][i] for k in cur}
        info = {"juv": stage0[i] == 1, "age": agents.age[i], "start": birth_energy[i]}
        records.append({**info, **life})
    new = ~alive0 & agents.alive
    births += int(new.sum())
    for k in cur:
        base[k][new] = cur[k][new]
    birth_energy[new] = agents.energy[new]
    if sim.tick % 2000 == 0:
        pops.append(int(agents.alive.sum()))

juv = [r for r in records if r["juv"]]
adult = [r for r in records if not r["juv"]]


def mean(rs: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rs])) if rs else float("nan")


feeds = int(stim["kin"].feed_events.sum())
share = len(juv) / max(births, 1)
print(
    f"{cond} seed {seed}: births {births}, juvenile deaths {len(juv)} ({share:.0%}),"
    f" adult deaths {len(adult)}, feeds {feeds}, pop {pops}"
)
for name, rs in (("juvenile", juv), ("adult", adult)):
    median = np.median([r["age"] for r in rs]) if rs else float("nan")
    print(
        f"  {name:8s} age {mean(rs, 'age'):6.0f} (median {median:.0f})"
        f"  start E {mean(rs, 'start'):5.1f}  food {mean(rs, 'eaten'):5.2f}"
        f"  harvest {mean(rs, 'harvest'):5.1f}  hazard dmg {mean(rs, 'hazard'):5.1f}"
        f"  poison dmg {mean(rs, 'poison'):5.1f}  brain invest {mean(rs, 'invest'):5.1f}"
        f"  from parent {mean(rs, 'received'):5.2f}"
    )
if juv:
    q = np.percentile([r["age"] for r in juv], [10, 25, 50, 75, 90]).round()
    print("  juvenile age at death, quantiles 10/25/50/75/90:", q)
