import math

import numpy as np
import pytest

from spiking_evo.snn.neuron import LIFPopulation


def spike_times(pop: LIFPopulation, i_ext: float, steps: int) -> list[int]:
    return [t for t in range(steps) if pop.step(i_ext=i_ext)[0]]


@pytest.mark.parametrize(
    ("i_ext", "tau", "thresh", "refr"),
    [(1.5, 20.0, 1.0, 2.0), (3.0, 10.0, 1.0, 0.0), (1.2, 30.0, 1.0, 5.0)],
)
def test_constant_input_fires_at_expected_rate(i_ext, tau, thresh, refr):
    dt = 0.1
    pop = LIFPopulation(1, tau_m=tau, v_thresh=thresh, v_reset=0.0, refractory_ms=refr, dt_ms=dt)
    steps = 50_000  # 5 s
    times = spike_times(pop, i_ext, steps)
    intervals = np.diff(times)

    # Exact discrete period: k integrating steps to cross threshold + refractory steps.
    decay = math.exp(-dt / tau)
    k = math.ceil(math.log(1 - thresh / i_ext) / math.log(decay))
    assert set(intervals.tolist()) == {k + round(refr / dt)}

    # Continuous-time LIF: T = t_ref + tau * ln(I / (I - theta)); within one dt.
    period_ms = refr + tau * math.log(i_ext / (i_ext - thresh))
    assert abs(intervals.mean() * dt - period_ms) <= dt
    rate_hz = len(times) / (steps * dt / 1000)
    assert rate_hz == pytest.approx(1000 / period_ms, rel=0.02)


def test_subthreshold_input_never_fires():
    pop = LIFPopulation(1, tau_m=20.0, v_thresh=1.0, dt_ms=1.0)
    assert spike_times(pop, 0.99, 5000) == []
    assert pop.v[0] == pytest.approx(0.99, abs=1e-6)


def test_refractory_ignores_input():
    pop = LIFPopulation(1, tau_m=20.0, v_thresh=1.0, refractory_ms=3.0, dt_ms=1.0)
    assert pop.step(i_syn=2.0)[0]
    for _ in range(3):
        assert not pop.step(i_syn=2.0)[0]
        assert pop.v[0] == 0.0
    assert pop.step(i_syn=2.0)[0]


def test_parameters_are_per_neuron():
    pop = LIFPopulation(2, tau_m=20.0, v_thresh=np.array([1.0, 2.0]), dt_ms=1.0)
    spikes = pop.step(i_syn=1.5)
    assert spikes.tolist() == [True, False]
