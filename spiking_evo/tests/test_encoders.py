import numpy as np
import pytest

from spiking_evo.snn.controller import SNNController
from spiking_evo.snn.encoders import CurrentEncoder, LatencyEncoder, RateEncoder, make_encoder
from spiking_evo.snn.network import NetworkBatch


def test_rate_encoder_mean_rate():
    enc = RateEncoder(max_rate_hz=200.0, dt_ms=1.0)
    x = np.array([[0.0, 0.5, 1.0]])
    spikes = enc.encode(x, 20_000, np.random.default_rng(0)).spikes
    rates = spikes.mean(axis=0)[0] * 1000.0
    assert rates[0] == 0.0
    assert rates[1] == pytest.approx(100.0, rel=0.05)
    assert rates[2] == pytest.approx(200.0, rel=0.05)


def test_rate_encoder_rejects_impossible_rate():
    with pytest.raises(ValueError):
        RateEncoder(max_rate_hz=2000.0, dt_ms=1.0)


def test_latency_encoder_earlier_for_stronger_input():
    x = np.array([[1.0, 0.5, 0.01, 0.0]])
    spikes = LatencyEncoder().encode(x, 10, np.random.default_rng(0)).spikes
    assert spikes.sum(axis=0).tolist() == [[1, 1, 1, 0]]
    t = spikes[:, 0, :3].argmax(axis=0)
    assert t.tolist() == [0, 4, 8]  # floor((1 - x) * 9)


def test_current_encoder():
    enc = CurrentEncoder(gain=2.0)
    out = enc.encode(np.array([[0.0, 0.5, 2.0]]), 10, np.random.default_rng(0))
    assert out.spikes is None
    assert out.current.tolist() == [[0.0, 1.0, 2.0]]


def test_make_encoder_unknown():
    with pytest.raises(ValueError):
        make_encoder("nope", max_rate_hz=1, dt_ms=1, current_gain=1)


def test_controller_decodes_motor_pairs():
    # Input 0 drives "forward" (output 0); input 1 drives "turn right" (output 3).
    net = NetworkBatch.empty(2, n_inputs=2, n_outputs=4, n_hidden=0, max_delay=1)
    net.weights[:, 0, 0, 2] = 2.0
    net.weights[:, 0, 1, 5] = 2.0
    ctrl = SNNController(
        net,
        LatencyEncoder(),
        dt_ms=1.0,
        steps_per_tick=10,
        bias_input=False,
        motor_threshold_hz=0.0,
        rng=np.random.default_rng(0),
    )
    fwd, turn = ctrl.act(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert fwd.tolist() == [1, 0]
    assert turn.tolist() == [0, 1]
    assert ctrl.noninput_spikes.tolist() == [1, 1]
    assert ctrl.mean_firing_rate_hz() == pytest.approx([25.0, 25.0])  # 1 spike/4 neurons/10ms
