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

from .neuron import decay_factor, refractory_steps


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
    # Plasticity (optional): which synapses may learn, and each network's learning rate.
    plastic: np.ndarray | None = None  # (B, D, N, N) bool
    learning_rate: np.ndarray | None = None  # (B,)

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
    """Batched LIF network with optional reward-modulated STDP.

    Plasticity (``plasticity`` given and ``net.plastic`` set): every neuron keeps an
    exponentially decaying spike trace x (time constant ``tau_trace_ms``). Each plastic
    synapse src -> dst keeps an eligibility e (time constant ``tau_eligibility_ms``):

        when dst spikes:  e += x[src]                    (src fired shortly before: causal)
        when src spikes:  e -= a_minus_ratio * x[dst]    (dst fired shortly before: acausal)

    ``apply_reward(r)`` turns eligibility into learning: w += learning_rate * r * e, clipped
    to +-weight_max. ``reset()`` restores the genome's weights (nothing learned is inherited).
    """

    def __init__(
        self,
        net: NetworkBatch,
        dt_ms: float,
        plasticity=None,
        weight_max: float = 5.0,
    ) -> None:
        self.net = net
        self.dt_ms = dt_ms
        b, d, n = net.batch, net.max_delay, net.n_neurons
        self._decay = decay_factor(net.tau_m, dt_ms)
        self._gain = 1.0 - self._decay  # weight of the external current
        self._ref_steps = refractory_steps(net.refractory_ms, dt_ms)
        self._thresh = np.where(net.active, net.v_thresh, np.inf)
        self._v_reset = np.ascontiguousarray(net.v_reset)
        # Evolved networks are sparse: propagate along the list of existing synapses
        # (gather from the spike history, scatter-add into the targets).
        learns = plasticity is not None and net.plastic is not None
        mask = (net.weights != 0) | net.plastic if learns else net.weights != 0
        bi, di, si, ti = np.nonzero(mask)
        self._syn_w0 = net.weights[bi, di, si, ti]
        self._syn_w = self._syn_w0.copy()
        self._syn_to = bi * n + ti  # flat index into (B, N)
        # The spike history is a ring buffer of D slots of shape (B, N); slot `ptr` is
        # written at the current step. For each value of ptr, precompute the flat index
        # of the slot holding the spikes emitted (delay) steps ago for every synapse.
        base = bi * n + si
        self._syn_from = np.stack([((ptr - 1 - di) % d) * (b * n) + base for ptr in range(d)])
        self.v = np.empty((b, n))
        self.ref = np.empty((b, n), dtype=np.int64)
        self._history = np.empty((d, b, n))
        self._hist_flat = self._history.reshape(-1)
        self._buf = np.empty((b, n))
        self._integrating = np.empty((b, n), dtype=bool)
        self._ptr = 0
        self.learns = learns
        if learns:
            pl = np.flatnonzero(net.plastic[bi, di, si, ti])
            self._pl = pl  # positions (in the synapse list) of the plastic synapses
            self._pl_pre = (bi * n + si)[pl]
            self._pl_post = (bi * n + ti)[pl]
            self._pl_batch = bi[pl]
            self._pl_lr = np.asarray(net.learning_rate, dtype=np.float64)[bi[pl]]
            self._trace_decay = np.exp(-dt_ms / plasticity.tau_trace_ms)
            self._elig_decay = np.exp(-dt_ms / plasticity.tau_eligibility_ms)
            self._a_minus = plasticity.a_minus_ratio
            self._wmax = weight_max
            self._trace = np.zeros(b * n)
            self._elig = np.zeros(pl.size)
        self.reset()

    def reset(self) -> None:
        np.copyto(self._syn_w, self._syn_w0)
        if self.learns:
            self._trace.fill(0.0)
            self._elig.fill(0.0)
        np.copyto(self.v, self._v_reset)
        self.ref.fill(0)
        self._history.fill(0.0)
        self._ptr = 0

    def step(self, i_ext: np.ndarray | None = None, forced: np.ndarray | None = None) -> np.ndarray:
        """One SNN step (same dynamics as ``neuron.lif_step``, with reused buffers).

        i_ext:  (B, N) constant external current for this step (or None).
        forced: (B, N) boolean spikes imposed from outside, e.g. by a spike encoder.
        Returns (B, N) boolean spikes emitted at this step.
        """
        d, b, n = self._history.shape
        contrib = self._hist_flat[self._syn_from[self._ptr]] * self._syn_w
        i_syn = np.bincount(self._syn_to, weights=contrib, minlength=b * n).reshape(b, n)

        v, ref, buf, integ = self.v, self.ref, self._buf, self._integrating
        np.less_equal(ref, 0, out=integ)
        np.multiply(v, self._decay, out=buf)
        if i_ext is not None:
            buf += self._gain * i_ext
        buf += i_syn
        np.copyto(v, buf, where=integ)
        np.invert(integ, out=integ)  # now: refractory neurons
        np.copyto(v, self._v_reset, where=integ)
        np.subtract(ref, 1, out=ref, where=integ)
        spikes = v >= self._thresh
        spikes &= ~integ  # refractory neurons never spike
        np.copyto(v, self._v_reset, where=spikes)
        np.copyto(ref, self._ref_steps, where=spikes)
        if forced is not None:
            spikes |= forced & self.net.active
        self._history[self._ptr] = spikes
        self._ptr = (self._ptr + 1) % d
        if self.learns and self._pl.size:
            sp = spikes.reshape(-1)
            post = sp[self._pl_post]
            pre = sp[self._pl_pre]
            self._elig *= self._elig_decay
            self._elig += post * self._trace[self._pl_pre]
            self._elig -= self._a_minus * pre * self._trace[self._pl_post]
            self._trace *= self._trace_decay
            self._trace += sp
        return spikes

    def apply_reward(self, reward: np.ndarray) -> None:
        """Convert eligibility into weight changes; ``reward`` is (B,) in about [-1, 1]."""
        if not self.learns or self._pl.size == 0:
            return
        w = self._syn_w[self._pl] + self._pl_lr * reward[self._pl_batch] * self._elig
        self._syn_w[self._pl] = np.clip(w, -self._wmax, self._wmax)

    def weight_change(self) -> np.ndarray:
        """(B,) mean absolute change of the plastic weights since the last reset."""
        b = self.net.batch
        if not self.learns or self._pl.size == 0:
            return np.zeros(b)
        diff = np.abs(self._syn_w[self._pl] - self._syn_w0[self._pl])
        total = np.bincount(self._pl_batch, weights=diff, minlength=b)
        count = np.bincount(self._pl_batch, minlength=b)
        return total / np.maximum(count, 1)
