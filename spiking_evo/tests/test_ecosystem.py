import csv
import json

import numpy as np
import pytest

from spiking_evo.config import config_from_dict
from spiking_evo.ecosystem import Ecosystem


def eco_config(**eco):
    base = {
        "experiment": "eco_test",
        "seed": 5,
        "world": {"width": 12, "height": 12},
        "agent": {"initial_energy": 40.0, "metabolic_cost": 0.05, "move_cost": 0.0},
        "stimuli": [{"type": "body"}, {"type": "food", "count": 40, "regrow_rate": 0.2,
                                       "energy_value": 10.0}],
        "network": {"hidden_max": 4, "hidden_init": 2, "steps_per_tick": 4},
        "ecosystem": {"capacity": 30, "initial_population": 10, "log_every": 20,
                      "init_repro_threshold": 30.0, "init_juvenile_age": 10.0,
                      "repro_cooldown": 5, "max_age": 400, **eco},
    }  # fmt: skip
    return config_from_dict(base)


def test_births_deaths_and_bookkeeping(tmp_path):
    sim = Ecosystem(eco_config(), tmp_path / "out")
    rows = sim.run(ticks=200)
    assert sum(r["births"] for r in rows) > 0
    a = sim.world.agents
    alive = np.flatnonzero(a.alive)
    assert alive.size <= 30
    kids = alive[sim.parent[alive] >= 0]
    assert kids.size and (sim.generation[kids] >= 1).all()
    # every living child's parent id was issued before its own id
    assert (sim.parent[kids] < sim.ids[kids]).all()
    with open(tmp_path / "out/ecosystem.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    snap = json.loads(sorted((tmp_path / "out/snapshots").iterdir())[-1].read_text())
    assert len(snap["agents"]) == alive.size


def test_same_seed_same_history():
    r1 = Ecosystem(eco_config()).run(ticks=120)
    r2 = Ecosystem(eco_config()).run(ticks=120)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows]  # noqa: E731
    assert strip(r1) == strip(r2)


def test_juveniles_are_slow_and_cannot_reproduce():
    sim = Ecosystem(eco_config(init_juvenile_age=1000.0, juvenile_speed=0.5))
    a = sim.world.agents
    a.age[:] = 0
    sim._update_stage()
    assert (a.stage[a.alive] == 1).all()
    a.energy[a.alive] = 1000.0  # rich but juvenile: no births
    x0 = a.x.copy()
    sim.step()
    assert sim._window["births"] == 0
    # on odd ages a juvenile may not move
    a.age[:] = 1
    x1, y1 = a.x.copy(), a.y.copy()
    sim.world.step(sim.wrapper)
    moved = (a.x != x1) | (a.y != y1)
    assert not moved[a.alive & (a.age == 2)].any()
    assert x0.shape == a.x.shape


def test_old_age_and_capacity():
    sim = Ecosystem(eco_config(max_age=50, capacity=12, initial_population=12))
    sim.world.agents.age[:] = 49
    sim.step()
    assert sim._window["deaths_old_age"] > 0
    sim2 = Ecosystem(eco_config(capacity=12, initial_population=12))
    sim2.world.agents.energy[:] = 1000.0
    sim2.world.agents.age[:] = 100
    sim2._update_stage()
    sim2.step()
    assert sim2.world.agents.alive.sum() <= 12 and sim2._window["births"] == 0


def test_traits_mutate_within_bounds():
    sim = Ecosystem(eco_config(trait_sigma=0.5, clutch_change=1.0))
    sim.traits.repro_threshold[0] = 400.0
    for c in range(1, 30):
        sim._inherit_traits(0, c)
    tr = sim.traits
    assert 20.0 <= tr.repro_threshold[1:].min() and tr.repro_threshold[1:].max() <= 500.0
    assert set(tr.clutch[1:].tolist()) <= {1, 2, 3}
    assert len(set(np.round(tr.endowment[1:], 6))) > 1


def test_newborn_networks_do_not_reset_living_ones():
    sim = Ecosystem(eco_config())
    for _ in range(5):
        sim.step()
    net = sim.controller.network
    v_before = net.v.copy()
    keep = np.ones(sim.K, dtype=bool)
    keep[-1] = False
    from spiking_evo.genetics.genome import build_network_batch

    sim.controller.replace_networks(build_network_batch(sim.genomes, sim.spec), keep)
    assert np.array_equal(sim.controller.network.v[:-1], v_before[:-1])


def test_init_from_seed_population(tmp_path):
    import gzip

    sim = Ecosystem(eco_config())
    src = {"format": "spiking_evo.checkpoint", "version": 1, "experiment": "x", "seed": 0,
           "next_generation": 1, "input_names": ["food.front", "bias"],
           "genomes": [g.to_dict() for g in sim.genomes[:3]]}  # fmt: skip
    # genomes with a different sensory layout get adapted
    for g in src["genomes"]:
        g["n_inputs"] = 2
        g["synapses"] = {"src": [], "dst": [], "weight": [], "delay": [], "plastic": []}
        nr = g["neurons"]
        cut = len(sim.input_names) - 2
        for k in nr:
            nr[k] = nr[k][cut:]
    path = tmp_path / "pop.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(src, f)
    sim2 = Ecosystem(eco_config(init_from=str(path)))
    assert all(g.n_inputs == len(sim2.input_names) for g in sim2.genomes)
    with pytest.raises(ValueError):
        eco_config(capacity="x")


def d2_config(**eco):
    cfg = eco_config(**eco)
    from spiking_evo.config import config_to_dict

    d = config_to_dict(cfg)
    d["stimuli"].append({"type": "kin", "feed_amount": 2.0, "feed_efficiency": 0.5})
    d["network"]["action_outputs"] = 1
    d["network"]["development"] = {"enabled": True, "trigger": "growth",
                                   "init_dormant_prob": 1.0, "init_wake_at": 3.0}  # fmt: skip
    d["ecosystem"]["growth_rate"] = 1.0
    d["ecosystem"]["growth_reserve"] = 5.0
    return config_from_dict(d)


def test_growth_wakes_neurons_by_investing_energy():
    sim = Ecosystem(d2_config())
    a = sim.world.agents
    i = int(np.flatnonzero(a.alive & (sim.controller.dormant_left() > 0))[0])
    left0 = sim.controller.dormant_left()[i]
    a.energy[i] = 100.0
    e0 = a.energy[i]
    sim._grow()
    assert a.energy[i] == e0 - 1.0  # invested growth_rate
    for _ in range(3):
        sim._grow()
    assert sim.controller.dormant_left()[i] < left0  # wake_at 3 reached
    a.energy[i] = 5.0  # at the reserve: no more investment
    sim._grow()
    assert a.energy[i] == 5.0


def test_kin_sense_and_feeding():
    sim = Ecosystem(d2_config())
    a, kin = sim.world.agents, {s.name: s for s in sim.world.stimuli}["kin"]
    a.alive[:] = False
    # adult 0 (id 100) at (5,5) facing north; its juvenile 1 just north; stranger 2 adjacent
    for slot, (x, y, stage, ident, par) in enumerate(
        [(5, 5, 0, 100, -1), (5, 4, 1, 101, 100), (6, 5, 1, 102, 999)]
    ):
        a.alive[slot], a.x[slot], a.y[slot], a.stage[slot] = True, x, y, stage
        sim.ids[slot], sim.parent[slot] = ident, par
    a.heading[0] = 0
    a.energy[:3] = [50.0, 1.0, 0.5]
    obs = kin.sense(a, sim.world)
    ch = kin.channels
    assert obs[0, ch.index("child_front")] > 0 and obs[0, ch.index("child_here")] == 1.0
    assert obs[0, ch.index("child_left")] == 0
    assert obs[1, ch.index("parent_here")] == 1.0
    a.action[:] = False
    a.action[0, 0] = True
    delta = kin.interact(a, sim.world)
    assert delta[0] == -2.0 and delta[1] == 1.0 and delta[2] == 0.0  # only own child, 50% lost
    kin.kin_only = False
    delta = kin.interact(a, sim.world)
    assert delta[2] == 1.0  # the poorest juvenile in reach
    a.stage[0] = 1  # juveniles cannot feed
    assert not kin.interact(a, sim.world).any()


def test_d2_runs_end_to_end(tmp_path):
    sim = Ecosystem(d2_config(), tmp_path)
    rows = sim.run(ticks=100)
    assert rows and rows[-1]["growth_invested"] >= 0 and rows[-1]["feed_events"] != ""
