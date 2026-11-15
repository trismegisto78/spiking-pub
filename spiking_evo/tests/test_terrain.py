import csv

import numpy as np
import pytest

from spiking_evo.config.schema import AgentConfig, WorldConfig
from spiking_evo.core.world import World
from spiking_evo.genetics.population import Evolution
from spiking_evo.snn.controller import SNNController
from spiking_evo.snn.encoders import LatencyEncoder
from spiking_evo.snn.network import NetworkBatch
from spiking_evo.stimuli.terrain import TerrainStimulus


class Scripted:
    """Stays still and acts according to a per-step schedule."""

    def __init__(self, schedule):
        self.schedule, self.t = schedule, 0

    def reset(self):
        self.t = 0

    def act(self, obs, active):
        n = obs.shape[0]
        do = self.schedule(self.t)
        self.t += 1
        return np.zeros(n, int), np.zeros(n, int), np.full((n, 1), do)


def make_world(terrain, n=1, arenas="isolated", wrap=True):
    cfg = WorldConfig(width=12, height=12, steps_per_episode=50, wrap=wrap, arenas=arenas)
    agent = AgentConfig(initial_energy=100.0, metabolic_cost=0.0, move_cost=0.0)
    w = World(cfg, agent, [terrain], n, n_actions=1)
    w.reset(0, "t")
    return w


def soil(terrain, fertile):
    terrain.fertile = np.full(terrain.sown.shape, fertile)


def put(w, x=5, y=5):
    w.agents.x[:], w.agents.y[:], w.agents.heading[:] = x, y, 0


def test_requires_an_action_output():
    cfg = WorldConfig(width=8, height=8)
    w = World(cfg, AgentConfig(), [TerrainStimulus()], 1)  # n_actions=0
    with pytest.raises(ValueError, match="action_outputs"):
        w.reset(0, "t")


def test_planting_on_fertile_soil_then_harvest():
    tr = TerrainStimulus(grow_time=3, harvest_energy=15.0, action_cost=0.5)
    w = make_world(tr)
    soil(tr, True)
    put(w)
    ctrl = Scripted(lambda t: t == 0)  # act once, then wait on the cell
    gains = []
    for _ in range(6):
        before = w.agents.energy[0]
        w.step(ctrl)
        gains.append(round(w.agents.energy[0] - before, 6))
    assert gains == [-0.5, 0.0, 0.0, 15.0, 0.0, 0.0]  # ripe 3 steps after planting
    st = tr.episode_stats()
    assert (st["planted"][0], st["harvested"][0], st["poisoned"][0]) == (1, 1, 0)


def test_acting_on_barren_soil_poisons():
    tr = TerrainStimulus(poison_damage=2.0, poison_duration=3, action_cost=0.5)
    w = make_world(tr)
    soil(tr, False)
    put(w)
    ctrl = Scripted(lambda t: t == 0)
    gains = []
    for _ in range(5):
        before = w.agents.energy[0]
        w.step(ctrl)
        gains.append(round(w.agents.energy[0] - before, 6))
    assert gains == [-2.5, -2.0, -2.0, 0.0, 0.0]  # poisoned for 3 steps
    assert tr.episode_stats()["poisoned"][0] == 1


def test_acting_again_on_occupied_cell_only_costs():
    tr = TerrainStimulus(grow_time=100, action_cost=0.5)
    w = make_world(tr)
    soil(tr, True)
    put(w)
    ctrl = Scripted(lambda t: True)
    for _ in range(4):
        w.step(ctrl)
    st = tr.episode_stats()
    assert st["planted"][0] == 1 and st["actions"][0] == 4
    assert w.agents.energy[0] == pytest.approx(100.0 - 4 * 0.5)


def test_consequences_are_sensed_but_soil_is_not():
    ch = ["crop_front", "crop_here", "poison_front", "poison_here"]
    tr = TerrainStimulus(grow_time=0, channels=ch, sense_gain=1.0)
    w = make_world(tr)
    put(w)
    assert tr.sense(w.agents, w).tolist() == [[0.0, 0.0, 0.0, 0.0]]  # fertile or not: no clue
    tr.sown[0, 4, 5] = 1  # ripe crop one cell north (front)
    tr.poison_until[0, 5, 5] = 10**6  # poison here
    obs = tr.sense(w.agents, w)[0]
    assert obs[0] == pytest.approx(1 - np.exp(-1.0)) and obs[1] == 0.0
    assert obs[2] == 0.0 and obs[3] == 1.0


def test_walled_border_reads_as_empty():
    tr = TerrainStimulus(grow_time=0, channels=["crop_front"])
    w = make_world(tr, wrap=False)
    put(w, x=5, y=0)  # facing north at the top wall
    assert tr.sense(w.agents, w)[0, 0] == 0.0


def test_dead_agents_do_not_act():
    tr = TerrainStimulus()
    w = make_world(tr, n=2)
    soil(tr, True)
    put(w)
    w.agents.alive[1] = False
    w.agents.action[:] = True
    delta = tr.interact(w.agents, w)
    assert delta[1] == 0.0 and tr.episode_stats()["actions"].tolist() == [1, 0]


def test_shared_arena_one_planter_per_cell():
    tr = TerrainStimulus()
    w = make_world(tr, n=3, arenas="shared")
    soil(tr, True)
    put(w)
    w.agents.action[:] = True
    tr.interact(w.agents, w)
    assert tr.episode_stats()["planted"].sum() == 1


def test_controller_decodes_action_neuron():
    net = NetworkBatch.empty(2, n_inputs=1, n_outputs=5, n_hidden=0, max_delay=1)
    net.weights[:, 0, 0, 5] = 2.0  # input -> action neuron (output index 4)
    ctrl = SNNController(
        net, LatencyEncoder(), dt_ms=1.0, steps_per_tick=10, bias_input=False,
        motor_threshold_hz=0.0, rng=np.random.default_rng(0), action_threshold_hz=50.0,
    )  # fmt: skip
    fwd, turn, act = ctrl.act(np.array([[1.0], [0.0]]))
    assert act.tolist() == [[True], [False]] and fwd.tolist() == [0, 0]


def test_evolution_with_terrain_logs_stats_and_is_chunk_invariant(tiny_config, tmp_path):
    cfg = tiny_config(
        network={"action_outputs": 1},
        stimuli=[
            {"type": "food", "count": 12, "regrow_rate": 0.05},
            {"type": "hazard", "count": 2, "damage": [1.0, 4.0], "shared_layout": False},
            {"type": "terrain", "fertile_zones": 2, "fertile_radius": 2.0, "grow_time": 10,
             "shared_layout": False},
        ],
        world={"same_start": False},
    )  # fmt: skip
    evo = Evolution(cfg)
    pop = evo.initial_population()
    assert evo.spec.n_outputs == 5 and "terrain.crop_here" in evo.input_names
    from spiking_evo.genetics.population import merge_evaluations

    full = evo.evaluate(pop, 2)
    parts = merge_evaluations(
        [evo.evaluate_chunk(pop[:5], 2, 0, 12), evo.evaluate_chunk(pop[5:], 2, 5, 12)]
    )
    assert np.array_equal(full.fitness, parts.fitness)
    for k, v in full.stimulus_stats.items():
        assert np.array_equal(v, parts.stimulus_stats[k]), k
    assert full.stimulus_stats["terrain.actions"].sum() > 0
    evo.run(out_root=tmp_path)
    with open(tmp_path / "tiny/3/stimuli.csv", newline="") as f:
        stats = {r["stat"] for r in csv.DictReader(f)}
    assert {"food.eaten", "hazard.steps_inside", "terrain.planted", "terrain.poisoned"} <= stats
