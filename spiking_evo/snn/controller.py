"""Glue between the world and the SNN: observations -> spikes -> motor actions.

Implements the ``Controller`` protocol of ``core.world``. Output neurons are read as two
antagonist pairs, decoded as firing rates over the tick window:

    output 0 / 1: forward / backward  -> move +1, 0 or -1 cells
    output 2 / 3: turn left / right   -> rotate -90, 0 or +90 degrees
"""

from __future__ import annotations

import numpy as np

from .encoders import EncodedInput
from .network import NetworkBatch, SpikingNetwork

N_MOTOR_OUTPUTS = 4


class SNNController:
    def __init__(
        self,
        net: NetworkBatch,
        encoder,
        *,
        dt_ms: float,
        steps_per_tick: int,
        bias_input: bool,
        motor_threshold_hz: float,
        rng: np.random.Generator,
    ) -> None:
        if net.n_outputs != N_MOTOR_OUTPUTS:
            raise ValueError(f"the controller needs {N_MOTOR_OUTPUTS} output neurons")
        self.net = net
        self.network = SpikingNetwork(net, dt_ms)
        self.encoder = encoder
        self.dt_ms = dt_ms
        self.steps_per_tick = steps_per_tick
        self.bias_input = bias_input
        self.motor_threshold_hz = motor_threshold_hz
        self.rng = rng
        b, n = net.batch, net.n_neurons
        self._i_ext = np.zeros((b, n))
        self._forced = np.zeros((b, n), dtype=bool)
        noninput = net.active.copy()
        noninput[:, : net.n_inputs] = False
        self._noninput = noninput
        self.n_noninput = noninput.sum(axis=1)
        self.noninput_spikes = np.zeros(b, dtype=np.int64)
        self.output_spikes = np.zeros((b, net.n_outputs), dtype=np.int64)
        self.snn_steps = 0

    def reset(self) -> None:
        self.network.reset()
        self.noninput_spikes.fill(0)
        self.output_spikes.fill(0)
        self.snn_steps = 0

    def act(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = self.net.batch
        x = observations
        if self.bias_input:
            x = np.concatenate([x, np.ones((b, 1))], axis=1)
        if x.shape != (b, self.net.n_inputs):
            raise ValueError(f"expected {self.net.n_inputs} input channels, got {x.shape[1]}")
        enc: EncodedInput = self.encoder.encode(x, self.steps_per_tick, self.rng)
        n_in = self.net.n_inputs
        i_ext = None
        if enc.current is not None:
            self._i_ext[:, :n_in] = enc.current
            i_ext = self._i_ext
        out = self.net.output_slice
        counts = np.zeros((b, self.net.n_outputs), dtype=np.int64)
        for t in range(self.steps_per_tick):
            forced = None
            if enc.spikes is not None:
                self._forced[:, :n_in] = enc.spikes[t]
                forced = self._forced
            spikes = self.network.step(i_ext=i_ext, forced=forced)
            counts += spikes[:, out]
            self.noninput_spikes += (spikes & self._noninput).sum(axis=1)
        self.output_spikes += counts
        self.snn_steps += self.steps_per_tick
        window_s = self.steps_per_tick * self.dt_ms / 1000.0
        rates = counts / window_s
        return self._decode(rates[:, 0] - rates[:, 1]), self._decode(rates[:, 3] - rates[:, 2])

    def _decode(self, diff: np.ndarray) -> np.ndarray:
        thr = self.motor_threshold_hz
        return np.where(diff > thr, 1, np.where(diff < -thr, -1, 0)).astype(np.int64)

    def mean_firing_rate_hz(self) -> np.ndarray:
        """Mean firing rate of the active non-input neurons, per network."""
        duration_s = self.snn_steps * self.dt_ms / 1000.0
        if duration_s == 0:
            return np.zeros(self.net.batch)
        return self.noninput_spikes / np.maximum(self.n_noninput, 1) / duration_s
