import json

import numpy as np
import pytest

from spiking_evo.config.schema import Config, MutationConfig
from spiking_evo.genetics.genome import Genome, GenomeSpec, build_network_batch
from spiking_evo.genetics.operators import (
    add_neuron,
    add_synapse,
    crossover,
    mutate,
    remove_neuron,
    remove_synapse,
    tournament_select,
)


@pytest.fixture
def spec():
    cfg = Config()
    cfg.network.hidden_max = 6
    cfg.network.hidden_init = 3
    cfg.network.max_delay = 4
    return GenomeSpec.from_config(cfg, n_inputs=3, n_outputs=4)


def assert_valid(g: Genome, spec: GenomeSpec):
    n = g.n_neurons
    assert spec.n_fixed <= n <= spec.n_max
    assert all(len(a) == n for a in (g.tau_m, g.v_thresh, g.v_reset, g.refractory_ms))
    assert len(g.src) == len(g.dst) == len(g.weight) == len(g.delay)
    assert ((g.src >= 0) & (g.src < n)).all()
    assert ((g.dst >= spec.n_inputs) & (g.dst < n)).all()
    assert (np.abs(g.weight) <= spec.weight_max).all()
    assert ((g.delay >= 1) & (g.delay <= spec.max_delay)).all()
    assert len(set(zip(g.src.tolist(), g.dst.tolist(), strict=True))) == g.n_synapses
    lo, hi = spec.thresh_range
    assert ((g.v_thresh >= lo) & (g.v_thresh <= hi)).all()


def test_random_genome_is_valid(spec):
    g = Genome.random(spec, np.random.default_rng(0))
    assert_valid(g, spec)
    assert g.n_hidden == 3 and g.n_synapses > 0


def test_repair_fixes_invalid_genome(spec):
    g = Genome.random(spec, np.random.default_rng(1))
    g.src = np.array([0, 0, 0, 99, 1, 2])
    g.dst = np.array([3, 3, 1, 4, 50, 5])  # duplicate, into input, missing src, missing dst
    g.weight = np.array([100.0, 1.0, 1.0, 1.0, 1.0, -100.0])
    g.delay = np.array([0, 2, 2, 2, 2, 9])
    g.v_thresh[:] = -3.0
    r = g.repair(spec)
    assert_valid(r, spec)
    assert r.src.tolist() == [0, 2] and r.dst.tolist() == [3, 5]
    assert r.weight.tolist() == [spec.weight_max, -spec.weight_max]
    assert r.delay.tolist() == [1, spec.max_delay]


def test_serialisation_roundtrip(spec):
    g = Genome.random(spec, np.random.default_rng(2))
    d = json.loads(json.dumps(g.to_dict()))
    assert Genome.from_dict(d).equals(g)
    d["version"] = 99
    with pytest.raises(ValueError):
        Genome.from_dict(d)


def test_mutation_always_valid_and_does_not_alias(spec):
    rng = np.random.default_rng(3)
    m = MutationConfig(
        weight=0.5,
        threshold=0.5,
        tau=0.5,
        delay=0.5,
        add_synapse=0.5,
        remove_synapse=0.5,
        add_neuron=0.5,
        remove_neuron=0.5,
        weight_sigma=3.0,
        threshold_sigma=3.0,
    )
    g = Genome.random(spec, rng)
    before = g.copy()
    sizes = set()
    for _ in range(300):
        g = mutate(g, spec, m, rng)
        assert_valid(g, spec)
        sizes.add(g.n_hidden)
    assert len(sizes) > 2  # the number of neurons evolves
    child = mutate(before, spec, m, rng)
    assert child is not before and before.equals(before.copy())


def test_structural_operators(spec):
    rng = np.random.default_rng(4)
    g = Genome.random(spec, rng)
    assert add_synapse(g, spec, rng).n_synapses == g.n_synapses + 1
    assert remove_synapse(g, spec, rng).n_synapses == g.n_synapses - 1
    grown = add_neuron(g, spec, rng)
    assert grown.n_hidden == g.n_hidden + 1 and grown.n_synapses == g.n_synapses + 1
    shrunk = remove_neuron(g, spec, rng)
    assert shrunk.n_hidden == g.n_hidden - 1
    for h in (grown, shrunk):
        assert_valid(h, spec)
    full = g
    while full.n_hidden < spec.hidden_max:
        full = add_neuron(full, spec, rng)
    assert add_neuron(full, spec, rng).n_hidden == spec.hidden_max


def test_remove_neuron_reindexes_synapses(spec):
    g = Genome.random(spec, np.random.default_rng(5))
    last = g.n_neurons - 1
    g = g.repair(spec)
    g.src = np.array([last, 0])
    g.dst = np.array([spec.n_inputs, last])
    g.weight = np.array([1.0, 2.0])
    g.delay = np.array([1, 1])
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = remove_neuron(g, spec, rng)
        assert_valid(h, spec)


def test_crossover_valid(spec):
    rng = np.random.default_rng(6)
    a, b = Genome.random(spec, rng), Genome.random(spec, rng)
    b = add_neuron(b, spec, rng)
    for _ in range(20):
        c = crossover(a, b, spec, rng)
        assert_valid(c, spec)
        assert c.n_neurons == a.n_neurons
        assert set(zip(c.src.tolist(), c.dst.tolist(), strict=True)) == set(
            zip(a.src.tolist(), a.dst.tolist(), strict=True)
        )


def test_tournament_selection():
    rng = np.random.default_rng(7)
    fitness = np.array([1.0, 5.0, 3.0, 2.0])
    assert (tournament_select(fitness, 4, 10, rng) == 1).all()
    picks = tournament_select(fitness, 1, 4000, rng)
    assert np.bincount(picks, minlength=4).min() > 800  # size 1 = uniform
    picks = tournament_select(fitness, 2, 4000, rng)
    assert np.bincount(picks, minlength=4)[1] > np.bincount(picks, minlength=4)[0]


def test_build_network_batch(spec):
    rng = np.random.default_rng(8)
    gs = [Genome.random(spec, rng), add_neuron(Genome.random(spec, rng), spec, rng)]
    net = build_network_batch(gs, spec)
    assert net.weights.shape == (2, spec.max_delay, spec.n_max, spec.n_max)
    for b, g in enumerate(gs):
        assert net.active[b].sum() == g.n_neurons
        assert np.count_nonzero(net.weights[b]) == np.count_nonzero(g.weight)
        assert net.weights[b, g.delay - 1, g.src, g.dst].tolist() == g.weight.tolist()
