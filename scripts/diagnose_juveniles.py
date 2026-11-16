"""Diagnose how juveniles and adults die in an ecosystem run (energy budget per life).

    python scripts/diagnose_juveniles.py SEED care|control TICKS [FEED_AMOUNT]

Uses spiking_evo/experiments/eco_d5_<cond>.yaml."""
import sys, numpy as np
from spiking_evo.config import load_config
from spiking_evo.ecosystem import Ecosystem
cfg = load_config(f"spiking_evo/experiments/eco_d5_{sys.argv[2]}.yaml"); cfg.seed = int(sys.argv[1])
if len(sys.argv) > 4:
    for x in cfg.stimuli:
        if x["type"] == "kin" and x.get("feed_amount", 2.0) > 0: x["feed_amount"] = float(sys.argv[4])
births = 0; pops = []
sim = Ecosystem(cfg); w = sim.world; a = w.agents; n = len(a)
st = {s.name: s for s in w.stimuli}
def snap():
    d = {"eaten": st["food"].eaten, "hazard": st["hazard"].damage_taken,
         "poison": st["terrain"].poison_steps * st["terrain"].poison_damage,
         "harvest": st["terrain"].harvest_energy_total, "received": st["kin"].energy_received,
         "gained": a.energy_gained}
    return {k: np.array(v, float).copy() for k, v in d.items()}
invest = np.zeros(n); orig = sim.controller.invest_in_growth
def hook(x):
    invest[:] += x; return orig(x)
sim.controller.invest_in_growth = hook
base = snap(); born_e = a.energy.copy(); base_inv = invest.copy()
rec = []
for t in range(int(sys.argv[3])):
    alive0, stage0 = a.alive.copy(), a.stage.copy()
    sim.step()
    cur = snap()
    died = alive0 & ~a.alive
    for i in np.flatnonzero(died):
        rec.append(dict(juv=stage0[i] == 1, age=a.age[i], start=born_e[i], invest=invest[i]-base_inv[i],
                        **{k: cur[k][i]-base[k][i] for k in cur}))
    new = ~alive0 & a.alive
    births += int(new.sum())
    if sim.tick % 2000 == 0: pops.append(int(a.alive.sum()))
    for k in cur: base[k][new] = cur[k][new]
    born_e[new] = a.energy[new]; base_inv[new] = invest[new]
J = [r for r in rec if r["juv"]]; A = [r for r in rec if not r["juv"]]
def m(rs, k): return np.mean([r[k] for r in rs]) if rs else float("nan")
print(f"{sys.argv[2]} seed {sys.argv[1]}: births {births}, deaths juvenile {len(J)} ({len(J)/max(births,1):.0%}), adult {len(A)}, feeds {int(st['kin'].feed_events.sum())}, pop {pops}")
for name, rs in (("juvenile", J), ("adult", A)):
    print(f"  {name:8s} age {m(rs,'age'):6.0f} (median {np.median([r['age'] for r in rs]) if rs else 0:.0f})  "
          f"start E {m(rs,'start'):5.1f}  food {m(rs,'eaten'):5.2f}  harvest {m(rs,'harvest'):5.1f}  "
          f"hazard dmg {m(rs,'hazard'):5.1f}  poison dmg {m(rs,'poison'):5.1f}  brain invest {m(rs,'invest'):5.1f}  "
          f"from parent {m(rs,'received'):5.2f}")
if J:
    ages = np.array([r["age"] for r in J]); print("  juvenile age at death quantiles 10/25/50/75/90:", np.percentile(ages,[10,25,50,75,90]).round())
