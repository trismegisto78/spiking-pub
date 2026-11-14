"""Leaky integrate-and-fire neuron, vectorised over arbitrary arrays of neurons.

Membrane dynamics between steps (resting potential 0, unit resistance):

    tau_m dv/dt = -v + I_ext

integrated exactly over one step of length dt, plus instantaneous synaptic jumps:

    v[t+1] = v[t] * exp(-dt/tau_m) + (1 - exp(-dt/tau_m)) * I_ext + I_syn[t]

When v reaches ``v_thresh`` the neuron emits a spike, v is set to ``v_reset`` and the
neuron ignores all input for ``refractory_steps`` steps. All parameters are per neuron.
"""

from __future__ import annotations

import numpy as np


def decay_factor(tau_m: np.ndarray | float, dt_ms: float) -> np.ndarray:
    return np.exp(-dt_ms / np.asarray(tau_m, dtype=np.float64))


def refractory_steps(refractory_ms: np.ndarray | float, dt_ms: float) -> np.ndarray:
    return np.rint(np.asarray(refractory_ms, dtype=np.float64) / dt_ms).astype(np.int64)


def lif_step(
    v: np.ndarray,
    ref: np.ndarray,
    i_syn: np.ndarray | float,
    i_ext: np.ndarray | float,
    decay: np.ndarray,
    v_thresh: np.ndarray,
    v_reset: np.ndarray,
    ref_steps: np.ndarray,
) -> np.ndarray:
    """Advance the neurons by one step, in place. Returns the boolean spike array."""
    integrating = ref <= 0
    v_new = v * decay + (1.0 - decay) * i_ext + i_syn
    np.copyto(v, np.where(integrating, v_new, v_reset))
    np.subtract(ref, 1, out=ref, where=~integrating)
    spikes = integrating & (v >= v_thresh)
    np.copyto(v, v_reset, where=spikes)
    np.copyto(ref, ref_steps, where=spikes)
    return spikes


class LIFPopulation:
    """Convenience wrapper holding the state of a group of independent LIF neurons."""

    def __init__(
        self,
        n: int,
        tau_m: float | np.ndarray = 20.0,
        v_thresh: float | np.ndarray = 1.0,
        v_reset: float | np.ndarray = 0.0,
        refractory_ms: float | np.ndarray = 2.0,
        dt_ms: float = 1.0,
    ) -> None:
        shape = (n,)
        self.decay = np.broadcast_to(decay_factor(tau_m, dt_ms), shape).copy()
        self.v_thresh = np.broadcast_to(np.asarray(v_thresh, float), shape).copy()
        self.v_reset = np.broadcast_to(np.asarray(v_reset, float), shape).copy()
        self.ref_steps = np.broadcast_to(refractory_steps(refractory_ms, dt_ms), shape).copy()
        self.v = self.v_reset.copy()
        self.ref = np.zeros(shape, dtype=np.int64)

    def step(self, i_ext: float | np.ndarray = 0.0, i_syn: float | np.ndarray = 0.0) -> np.ndarray:
        return lif_step(
            self.v, self.ref, i_syn, i_ext, self.decay, self.v_thresh, self.v_reset, self.ref_steps
        )
