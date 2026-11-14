import numpy as np
import pytest

from spiking_evo.snn.network import NetworkBatch, SpikingNetwork


def two_neuron_net(weight: float, delay: int, batch: int = 1) -> NetworkBatch:
    net = NetworkBatch.empty(batch, n_inputs=1, n_outputs=1, n_hidden=0, max_delay=8)
    net.weights[:, delay - 1, 0, 1] = weight
    return net


def run(net: NetworkBatch, steps: int, force_at: dict[int, np.ndarray]) -> np.ndarray:
    sim = SpikingNetwork(net, dt_ms=1.0)
    out = []
    for t in range(steps):
        forced = np.zeros((net.batch, net.n_neurons), dtype=bool)
        if t in force_at:
            forced[:, 0] = force_at[t]
        out.append(sim.step(forced=forced))
    return np.array(out)  # (T, B, N)


@pytest.mark.parametrize("delay", [1, 2, 5, 8])
def test_single_synapse_propagates_with_delay(delay):
    spikes = run(two_neuron_net(2.0, delay), 20, {3: np.array([True])})
    assert np.flatnonzero(spikes[:, 0, 0]).tolist() == [3]
    assert np.flatnonzero(spikes[:, 0, 1]).tolist() == [3 + delay]


def test_weak_synapse_does_not_fire_target():
    spikes = run(two_neuron_net(0.5, 2), 20, {3: np.array([True])})
    assert not spikes[:, 0, 1].any()


def test_inhibition_cancels_excitation():
    net = NetworkBatch.empty(1, n_inputs=2, n_outputs=1, n_hidden=0, max_delay=3)
    net.weights[0, 1, 0, 2] = 1.5  # excitatory, delay 2
    net.weights[0, 1, 1, 2] = -1.0  # inhibitory, delay 2
    sim = SpikingNetwork(net, dt_ms=1.0)
    fired = []
    for t in range(10):
        forced = np.zeros((1, 3), dtype=bool)
        if t == 1:
            forced[0, :2] = True
        fired.append(sim.step(forced=forced)[0, 2])
    assert not any(fired)


def test_batch_members_are_independent():
    net = two_neuron_net(2.0, 3, batch=3)
    net.weights[1] = 0.0
    net.active[2, 1] = False
    spikes = run(net, 10, {0: np.array([True, True, True])})
    assert np.flatnonzero(spikes[:, 0, 1]).tolist() == [3]
    assert not spikes[:, 1, 1].any()
    assert not spikes[:, 2, 1].any()


def test_reset_clears_state():
    net = two_neuron_net(2.0, 4)
    sim = SpikingNetwork(net, dt_ms=1.0)
    forced = np.zeros((1, 2), dtype=bool)
    forced[0, 0] = True
    sim.step(forced=forced)
    sim.reset()
    assert not any(sim.step()[0, 1] for _ in range(10))
