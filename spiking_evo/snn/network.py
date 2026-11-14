"""Batched spiking network with per-synapse integer delays.

A batch of B independent networks is simulated at once. Every network is padded to the
same number of neurons N; unused neurons are marked inactive (infinite threshold, no
synapses). Neuron layout inside each network is fixed:

    [0, n_inputs)                      input neurons (driven by the encoder)
    [n_inputs, n_inputs + n_outputs)   output (motor) neurons
    [n_inputs + n_outputs, N)          hidden neurons

``weights[b, d - 1, src, dst]`` is the weight of the synapse src -> dst with delay d (in
SNN steps). A spike emitted by ``src`` at step t is delivered to ``dst`` at step t + d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import decay_factor, lif_step, refractory_steps


@dataclass
class NetworkBatch:
    weights: np.ndarray  # (B, D, N, N)
    tau_m: np.ndarray  # (B, N)
    v_thresh: np.ndarray  # (B, N); inf for inactive neurons
    v_reset: np.ndarray  # (B, N)
    refractory_ms: np.ndarray  # (B, N)
    active: np.ndarray  # (B, N) bool
    n_inputs: int
    n_outputs: int

    @property
    def batch(self) -> int:
        return self.weights.shape[0]

    @property
    def max_delay(self) -> int:
        return self.weights.shape[1]

    @property
    def n_neurons(self) -> int:
        return self.weights.shape[2]

    @property
    def output_slice(self) -> slice:
        return slice(self.n_inputs, self.n_inputs + self.n_outputs)

    @classmethod
    def empty(
        cls,
        batch: int,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        max_delay: int,
        tau_m: float = 20.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        refractory_ms: float = 2.0,
    ) -> NetworkBatch:
        n = n_inputs + n_outputs + n_hidden
        full = lambda val: np.full((batch, n), val, dtype=np.float64)  # noqa: E731
        return cls(
            weights=np.zeros((batch, max_delay, n, n), dtype=np.float64),
            tau_m=full(tau_m),
            v_thresh=full(v_thresh),
            v_reset=full(v_reset),
            refractory_ms=full(refractory_ms),
            active=np.ones((batch, n), dtype=bool),
            n_inputs=n_inputs,
            n_outputs=n_outputs,
        )


class SpikingNetwork:
    def __init__(self, net: NetworkBatch, dt_ms: float) -> None:
        self.net = net
        self.dt_ms = dt_ms
        b, d, n = net.batch, net.max_delay, net.n_neurons
        self._decay = decay_factor(net.tau_m, dt_ms)
        self._ref_steps = refractory_steps(net.refractory_ms, dt_ms)
        self._thresh = np.where(net.active, net.v_thresh, np.inf)
        # Evolved networks are sparse: propagate along the list of existing synapses
        # (gather from the spike history, scatter-add into the targets).
        bi, di, si, ti = np.nonzero(net.weights)
        self._syn_w = net.weights[bi, di, si, ti]
        self._syn_from = (bi * d + di) * n + si  # flat index into history (B, D, N)
        self._syn_to = bi * n + ti  # flat index into (B, N)
        self.v = np.empty((b, n))
        self.ref = np.empty((b, n), dtype=np.int64)
        # history[:, k] holds the spikes emitted k + 1 steps ago.
        self.history = np.empty((b, d, n))
        self.reset()

    def reset(self) -> None:
        np.copyto(self.v, self.net.v_reset)
        self.ref.fill(0)
        self.history.fill(0.0)

    def step(self, i_ext: np.ndarray | None = None, forced: np.ndarray | None = None) -> np.ndarray:
        """One SNN step.

        i_ext:  (B, N) constant external current for this step (or None).
        forced: (B, N) boolean spikes imposed from outside, e.g. by a spike encoder.
        Returns (B, N) boolean spikes emitted at this step.
        """
        b, d, n = self.history.shape
        contrib = self.history.reshape(-1)[self._syn_from] * self._syn_w
        i_syn = np.bincount(self._syn_to, weights=contrib, minlength=b * n).reshape(b, n)
        spikes = lif_step(
            self.v,
            self.ref,
            i_syn,
            0.0 if i_ext is None else i_ext,
            self._decay,
            self._thresh,
            self.net.v_reset,
            self._ref_steps,
        )
        if forced is not None:
            spikes |= forced & self.net.active
        if d > 1:
            self.history[:, 1:] = self.history[:, :-1]
        self.history[:, 0] = spikes
        return spikes
