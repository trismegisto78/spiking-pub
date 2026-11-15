import json

import numpy as np
import pytest

from spiking_evo.config.schema import PlasticityConfig
from spiking_evo.genetics.genome import Genome
from spiking_evo.genetics.population import Evolution, merge_evaluations
from spiking_evo.snn.controller import SNNController
from spiking_evo.snn.encoders import LatencyEncoder
from spiking_evo.snn.network import NetworkBatch, SpikingNetwork

P = PlasticityConfig(enabled=True, tau_trace_ms=20.0, tau_eligibility_ms=1000.0)


def pair_net(weight=2.0, plastic=True, lr=0.5):
    """Neuron 0 (input) -> neuron 1 (output), delay 1."""
    net = NetworkBatch.empty(1, n_inputs=1, n_outputs=1, n_hidden=0, max_delay=2)
    net.weights[0, 0, 0, 1] = weight
    net.plastic = np.zeros(net.weights.shape, dtype=bool)
    net.plastic[0, 0, 0, 1] = plastic
    net.learning_rate = np.array([lr])
    return SpikingNetwork(net, dt_ms=1.0, plasticity=P, weight_max=5.0)


def drive(sim, force_steps, steps=10, i_post=None):
    for t in range(steps):
        forced = np.zeros((1, 2), dtype=bool)
        if t in force_steps:
            forced[0, force_steps[t]] = True
        sim.step(forced=forced)


def test_causal_pairing_is_potentiated_by_reward_and_depressed_by_punishment():
    for reward, sign in ((1.0, 1), (-1.0, -1)):
        sim = pair_net()
        drive(sim, {0: 0})  # input fires at 0, output fires at 1 through the synapse
        assert sim._elig[0] > 0
        w0 = sim._syn_w[0]
        sim.apply_reward(np.array([reward]))
        assert np.sign(sim._syn_w[0] - w0) == sign
        assert sim.weight_change()[0] > 0
        sim.reset()
        assert sim._syn_w[0] == w0 and sim._elig[0] == 0  # nothing learned is inherited


def test_acausal_pairing_gives_negative_eligibility():
    sim = pair_net(weight=0.1)  # too weak to make the output fire
    drive(sim, {0: 1, 3: 0})  # output forced first, input after
    assert sim._elig[0] < 0


def test_non_plastic_synapse_and_zero_rate_do_not_learn():
    for sim in (pair_net(plastic=False), pair_net(lr=0.0)):
        drive(sim, {0: 0})
        w0 = sim._syn_w.copy()
        sim.apply_reward(np.array([1.0]))
        assert np.array_equal(sim._syn_w, w0)


def test_reward_is_a_prediction_error():
    net = NetworkBatch.empty(1, n_inputs=1, n_outputs=4, n_hidden=0, max_delay=1)
    net.plastic = np.zeros(net.weights.shape, dtype=bool)
    net.learning_rate = np.array([0.1])
    p = PlasticityConfig(enabled=True, reward_scale=1.0, reward_baseline_steps=5.0)
    ctrl = SNNController(
        net,
        LatencyEncoder(),
        dt_ms=1.0,
        steps_per_tick=5,
        bias_input=False,
        motor_threshold_hz=0.0,
        rng=np.random.default_rng(0),
        plasticity=p,
    )
    seen = []
    ctrl.network.apply_reward = lambda r: seen.append(float(r[0]))
    for _ in range(40):
        ctrl.feedback(np.array([0.5]), np.array([True]))
    assert seen[0] == pytest.approx(0.5) and abs(seen[-1]) < 0.01  # expected reward: no signal


def plastic_config(tiny_config, **kw):
    return tiny_config(
        network={"plasticity": {"enabled": True, "init_learning_rate": 0.2}},
        genetics={"mutation": {"learning_rate": 0.5, "plastic_flip": 0.1}},
        **kw,
    )


def test_evolution_with_plasticity(tiny_config, tmp_path):
    cfg = plastic_config(tiny_config)
    evo = Evolution(cfg)
    pop = evo.initial_population()
    assert all(g.learning_rate == 0.2 for g in pop)
    assert 0 < np.mean([g.plastic.mean() for g in pop]) < 1
    ev = evo.evaluate(pop, 0)
    assert ev.stimulus_stats["brain.weight_change"].max() > 0
    parts = merge_evaluations(
        [evo.evaluate_chunk(pop[:5], 0, 0, 12), evo.evaluate_chunk(pop[5:], 0, 5, 12)]
    )
    assert np.array_equal(ev.fitness, parts.fitness)
    final = evo.run(out_root=tmp_path)
    assert len({g.learning_rate for g in final}) > 1  # the learning rate evolves
    # serialisation keeps the plasticity genes; version-1 genomes still load
    g = final[0]
    assert Genome.from_dict(json.loads(json.dumps(g.to_dict()))).equals(g)
    d = g.to_dict()
    d["version"] = 1
    del d["learning_rate"], d["synapses"]["plastic"]
    old = Genome.from_dict(d)
    assert old.learning_rate == 0.0 and not old.plastic.any()


def test_learning_changes_behaviour_compared_with_frozen_weights(tiny_config):
    cfg = plastic_config(tiny_config)
    pop = Evolution(cfg).initial_population()
    learning = Evolution(cfg).evaluate(pop, 1)
    cfg.network.plasticity.enabled = False
    frozen_evo = Evolution(cfg)
    frozen = frozen_evo.evaluate([g.repair(frozen_evo.spec) for g in pop], 1)
    assert not np.array_equal(learning.fitness, frozen.fitness)
