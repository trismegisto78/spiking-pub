import json

import numpy as np

from spiking_evo.config.schema import DevelopmentConfig, PlasticityConfig
from spiking_evo.genetics.genome import Genome
from spiking_evo.genetics.population import Evolution, merge_evaluations
from spiking_evo.snn.controller import SNNController
from spiking_evo.snn.encoders import LatencyEncoder
from spiking_evo.snn.network import NetworkBatch


def dev_net():
    """input 0 -> hidden 5 (dormant, wakes at difficulty 3) -> output 1 (forward)."""
    net = NetworkBatch.empty(1, n_inputs=1, n_outputs=4, n_hidden=1, max_delay=1)
    net.weights[0, 0, 0, 5] = 2.0
    net.weights[0, 0, 5, 1] = 2.0
    net.dormant = np.zeros((1, 6), dtype=bool)
    net.dormant[0, 5] = True
    net.wake_at = np.full((1, 6), 3.0)
    return net


def controller(net, dev=None, plasticity=None):
    return SNNController(
        net, LatencyEncoder(), dt_ms=1.0, steps_per_tick=10, bias_input=False,
        motor_threshold_hz=0.0, rng=np.random.default_rng(0),
        development=dev or DevelopmentConfig(enabled=True, neuron_cost=0.01),
        plasticity=plasticity,
    )  # fmt: skip


def test_dormant_neuron_wakes_with_difficulty_and_costs_energy():
    ctrl = controller(dev_net())
    alive = np.array([True])
    fwd, _ = ctrl.act(np.array([[1.0]]), alive)
    assert fwd[0] == 0  # the relay neuron is asleep: no movement
    assert ctrl.energy_cost()[0] == 4 * 0.01  # 4 outputs awake, hidden asleep
    ctrl.feedback(np.array([-2.0]), alive, np.array([-2.0]))  # difficulty 2 < 3
    assert ctrl.woken[0] == 0
    ctrl.feedback(np.array([-1.5]), alive, np.array([-1.5]))  # difficulty 3.5 >= 3
    assert ctrl.woken[0] == 1 and ctrl.energy_cost()[0] == 5 * 0.01
    fwd, _ = ctrl.act(np.array([[1.0]]), alive)
    assert fwd[0] == 1  # now the signal reaches the forward neuron
    ctrl.reset()  # a new life starts asleep again
    assert ctrl.woken[0] == 0 and ctrl.energy_cost()[0] == 4 * 0.01


def test_gains_do_not_count_as_difficulty():
    ctrl = controller(dev_net())
    ctrl.feedback(np.array([10.0]), np.array([True]), np.array([10.0]))
    assert ctrl.difficulty[0] == 0.0


def test_plastic_new_only_restricts_learning_to_woken_neurons():
    net = dev_net()
    net.weights[0, 0, 0, 2] = 1.0  # an old synapse not touching the dormant neuron
    net.plastic = net.weights != 0
    net.learning_rate = np.array([0.1])
    dev = DevelopmentConfig(enabled=True, plastic_new_only=True)
    ctrl = controller(net, dev, PlasticityConfig(enabled=True))
    assert ctrl.network._pl.size == 2  # only 0->5 and 5->1


def dev_config(tiny_config, **kw):
    return tiny_config(
        network={"development": {"enabled": True, "init_dormant_prob": 0.6,
                                 "init_wake_at": 0.5, "neuron_cost": 0.001}},
        genetics={"mutation": {"dormant_flip": 0.3, "wake_at": 0.5}},
        stimuli=[{"type": "food", "count": 12}, {"type": "hazard", "count": 3, "damage": 2.0}],
        **kw,
    )  # fmt: skip


def test_evolution_with_development(tiny_config, tmp_path):
    cfg = dev_config(tiny_config)
    evo = Evolution(cfg)
    pop = evo.initial_population()
    assert any(g.dormant.any() for g in pop)
    assert not any(g.dormant[: evo.spec.n_fixed].any() for g in pop)
    ev = evo.evaluate(pop, 0)
    st = ev.stimulus_stats
    assert st["brain.woken"].sum() > 0 and st["brain.difficulty"].max() > 0
    parts = merge_evaluations(
        [evo.evaluate_chunk(pop[:4], 0, 0, 12), evo.evaluate_chunk(pop[4:], 0, 4, 12)]
    )
    assert np.array_equal(ev.fitness, parts.fitness)
    final = evo.run(out_root=tmp_path)
    g = final[0]
    assert Genome.from_dict(json.loads(json.dumps(g.to_dict()))).equals(g)
    assert len({tuple(np.round(x.wake_at, 6)) for x in final}) > 1  # wake thresholds evolve
