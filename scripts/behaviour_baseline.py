"""Behavioural tests of the food-only baseline: what strategy did evolution find?

    python -m spiking_evo sweep spiking_evo/experiments/food_only_baseline.yaml --seeds 1-8
    python scripts/behaviour_baseline.py runs/food_only_baseline [out.json]

Takes the best genome of generation 299 of every seed and evaluates it on fresh episodes
(40 copies x 5 episodes, each copy in its own arena with a different start).

Conditions (sensor manipulations applied to the observations before the SNN):
  full      unchanged
  blind     all food channels zeroed (bias kept)
  no_dir    front/left/right zeroed, 'here' kept
  mirror    left and right swapped
Baselines: random walker, and a hand-written steer-toward-food rule.
Steering stats in 'full': action given which sector has the strongest food signal.
"""

import json
import sys
from pathlib import Path

import numpy as np

from spiking_evo.config import load_config
from spiking_evo.core.clock import Clock
from spiking_evo.core.world import World
from spiking_evo.genetics.genome import Genome, build_network_batch
from spiking_evo.genetics.population import Evolution
from spiking_evo.seeding import derive_rng
from spiking_evo.snn.controller import SNNController
from spiking_evo.snn.encoders import make_encoder
from spiking_evo.stimuli.registry import build_stimuli

CFG = load_config("spiking_evo/experiments/food_only_baseline.yaml")
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/food_only_baseline")
COPIES = 40  # agents per episode (each in its own arena, different start)
EPISODES = 5
F, L, R, H = 0, 1, 2, 3  # channel order: front, left, right, here
MODES = ("full", "blind", "no_dir", "mirror")


class Wrapped:
    def __init__(self, inner, mode, log=None):
        self.inner, self.mode, self.log = inner, mode, log

    def reset(self):
        self.inner.reset()

    def act(self, obs, active):
        o = obs.copy()
        if self.mode == "blind":
            o[:] = 0
        elif self.mode == "no_dir":
            o[:, [F, L, R]] = 0
        elif self.mode == "mirror":
            o[:, [L, R]] = o[:, [R, L]]
        fwd, turn = self.inner.act(o, active)
        if self.log is not None:
            self.log.append((obs[active], fwd[active], turn[active]))
        return fwd, turn


class RandomWalker:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        pass

    def act(self, obs, active):
        n = len(obs)
        return self.rng.integers(-1, 2, n), self.rng.integers(-1, 2, n)


class Steer:
    """Reference rule: turn toward the stronger lateral food signal, else go forward."""

    def reset(self):
        pass

    def act(self, obs, active):
        f, l_, r = obs[:, F], obs[:, L], obs[:, R]
        turn = np.where((l_ > f) & (l_ > r), -1, np.where((r > f) & (r > l_), 1, 0))
        return np.ones(len(obs), dtype=np.int64), turn


def world(n):
    return World(
        CFG.world,
        CFG.agent,
        build_stimuli(CFG.stimuli),
        n_agents=n,
        clock=Clock(CFG.network.dt_ms, CFG.network.steps_per_tick),
    )


def run(make_controller, label_seed):
    gained, cells, alive = [], [], []
    for ep in range(EPISODES):
        w = world(COPIES)
        keys = ("behaviour", label_seed, ep)
        res = w.run_episode(make_controller(keys), 777, *keys)
        gained.append(res.energy_gained)
        cells.append(res.visited_cells)
        alive.append(res.alive)
    return (
        float(np.concatenate(gained).mean()),
        float(np.concatenate(cells).mean()),
        float(np.concatenate(alive).mean()),
    )


def snn_factory(net, mode, log=None):
    enc = make_encoder(
        CFG.network.encoder,
        max_rate_hz=CFG.network.max_rate_hz,
        dt_ms=CFG.network.dt_ms,
        current_gain=CFG.network.current_gain,
    )

    def make(keys):
        c = SNNController(
            net,
            enc,
            dt_ms=CFG.network.dt_ms,
            steps_per_tick=CFG.network.steps_per_tick,
            bias_input=CFG.network.bias_input,
            motor_threshold_hz=CFG.network.motor_threshold_hz,
            rng=derive_rng(777, *keys, "encoder"),
        )
        return Wrapped(c, mode, log)

    return make


def steering(log):
    obs = np.concatenate([o for o, _, _ in log])
    fwd = np.concatenate([f for _, f, _ in log])
    turn = np.concatenate([t for _, _, t in log])
    d = obs[:, [F, L, R]]
    none = d.max(axis=1) == 0
    best = d.argmax(axis=1)
    out = {}
    for name, mask in [
        ("nessun cibo", none),
        ("cibo davanti", ~none & (best == 0)),
        ("cibo a sinistra", ~none & (best == 1)),
        ("cibo a destra", ~none & (best == 2)),
    ]:
        out[name] = dict(
            n=int(mask.sum()),
            avanti=float((fwd[mask] == 1).mean()),
            indietro=float((fwd[mask] == -1).mean()),
            gira_sx=float((turn[mask] == -1).mean()),
            gira_dx=float((turn[mask] == 1).mean()),
        )
    return out


def structure(genome):
    s = genome["synapses"]
    names = {
        0: "front",
        1: "left",
        2: "right",
        3: "here",
        4: "bias",
        5: "AVANTI",
        6: "INDIETRO",
        7: "GIRA_SX",
        8: "GIRA_DX",
    }
    direct = [
        f"{names[a]}->{names[b]} {w:+.2f}"
        for a, b, w in zip(s["src"], s["dst"], s["weight"], strict=True)
        if a < 5 and 5 <= b < 9 and abs(w) > 0.3
    ]
    return direct


evo = Evolution(CFG)
results = {}
seeds = sorted((p.name for p in ROOT.iterdir() if p.is_dir()), key=int)
for seed in seeds:
    data = json.loads((ROOT / seed / "snapshots" / "gen_0299.json").read_text())
    g = Genome.from_dict(data["genome"]).repair(evo.spec)
    net = build_network_batch([g] * COPIES, evo.spec)
    row = {}
    log = []
    for mode in MODES:
        row[mode] = run(snn_factory(net, mode, log if mode == "full" else None), int(seed))
    row["steering"] = steering(log)
    row["direct"] = structure(data["genome"])
    row["synapses"] = len(data["genome"]["synapses"]["src"])
    results[seed] = row
    st = row["steering"]
    print(
        f"seed {seed}: "
        + "  ".join(f"{m} {row[m][0]:.0f}" for m in MODES)
        + f"  | forward {st['nessun cibo']['avanti']:.2f}"
        + f"  food left -> turn L {st['cibo a sinistra']['gira_sx']:.2f}"
        + f" R {st['cibo a sinistra']['gira_dx']:.2f}"
        + f"  | {row['synapses']} syn: {', '.join(row['direct'])}",
        flush=True,
    )

rand = run(lambda keys: RandomWalker(derive_rng(777, *keys, "rand")), 0)
steer = run(lambda keys: Steer(), 0)
results["_baselines"] = {"random": rand, "steer_rule": steer}
print(f"random walker {rand[0]:.0f}  steer-toward-food rule {steer[0]:.0f}")
Path(sys.argv[2] if len(sys.argv) > 2 else "behaviour.json").write_text(
    json.dumps(results, indent=1)
)
