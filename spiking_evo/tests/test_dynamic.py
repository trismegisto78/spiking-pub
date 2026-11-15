import numpy as np

from spiking_evo.config.schema import AgentConfig, PlasticityConfig, WorldConfig
from spiking_evo.core.world import World
from spiking_evo.genetics.population import Evolution, merge_evaluations
from spiking_evo.snn.controller import SNNController
from spiking_evo.snn.encoders import LatencyEncoder
from spiking_evo.snn.network import NetworkBatch
from spiking_evo.stimuli.registry import build_stimuli


def world(specs, n=2, n_actions=1):
    cfg = WorldConfig(width=20, height=20)
    w = World(cfg, AgentConfig(), build_stimuli(specs), n, n_actions=n_actions)
    w.reset(0, "t")
    return w


def test_hazards_move_every_n_steps():
    w = world([{"type": "hazard", "count": 4, "relocate_every": 3}])
    hz = w.stimuli[0]
    grids = []
    for t in range(7):
        hz.step(w, t)
        grids.append(np.array(hz.grid[0]))
    assert np.array_equal(grids[0], grids[2]) and not np.array_equal(grids[2], grids[3])
    assert np.array_equal(grids[3], grids[5]) and not np.array_equal(grids[5], grids[6])


def test_terrain_soil_moves_and_crops_on_changed_soil_die():
    w = world([{"type": "terrain", "fertile_zones": 3, "relocate_every": 5}])
    tr = w.stimuli[0]
    tr.sown[:] = 1.0  # a crop everywhere
    before = np.array(tr.variety)
    for t in range(6):
        tr.step(w, t)
    changed = np.array(tr.variety) != before
    assert changed.any()
    assert (tr.sown[changed] == 0).all() and (tr.sown[~changed] == 1).all()


def test_stimuli_only_reward_ignores_the_cost_of_living():
    net = NetworkBatch.empty(1, n_inputs=1, n_outputs=4, n_hidden=0, max_delay=1)
    net.plastic = np.zeros(net.weights.shape, dtype=bool)
    net.learning_rate = np.array([0.1])
    seen = {}
    for mode in ("energy", "stimuli"):
        p = PlasticityConfig(enabled=True, reward=mode, reward_scale=1.0)
        ctrl = SNNController(
            net, LatencyEncoder(), dt_ms=1.0, steps_per_tick=5, bias_input=False,
            motor_threshold_hz=0.0, rng=np.random.default_rng(0), plasticity=p,
        )  # fmt: skip
        got = []
        ctrl.network.apply_reward = lambda r, got=got: got.append(float(r[0]))
        ctrl.feedback(np.array([-0.05]), np.array([True]), np.array([0.0]))
        seen[mode] = got[0]
    assert seen["energy"] < 0 and seen["stimuli"] == 0.0


def test_dynamic_world_is_chunk_invariant(tiny_config):
    cfg = tiny_config(
        network={"action_outputs": 1, "plasticity": {"enabled": True, "reward": "stimuli"}},
        world={"same_start": False},
        stimuli=[
            {"type": "food", "count": 12, "regrow_rate": 0.05, "shared_layout": False},
            {"type": "hazard", "count": 2, "relocate_every": 20, "shared_layout": False},
            {"type": "terrain", "fertile_zones": 2, "grow_time": 5, "relocate_every": 25,
             "shared_layout": False},
        ],
    )  # fmt: skip
    evo = Evolution(cfg)
    pop = evo.initial_population()
    full = evo.evaluate(pop, 2)
    parts = merge_evaluations(
        [evo.evaluate_chunk(pop[:5], 2, 0, 12), evo.evaluate_chunk(pop[5:], 2, 5, 12)]
    )
    assert np.array_equal(full.fitness, parts.fitness)
