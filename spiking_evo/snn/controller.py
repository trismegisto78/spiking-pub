"""Glue between the world and the SNN: observations -> spikes -> motor actions.

Implements the ``Controller`` protocol of ``core.world``. Output neurons are read as two
antagonist pairs, decoded as firing rates over the tick window:

    output 0 / 1: forward / backward  -> move +1, 0 or -1 cells
    output 2 / 3: turn left / right   -> rotate -90, 0 or +90 degrees
    outputs 4..:  generic actions      -> True when the rate exceeds action_threshold_hz
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
        action_threshold_hz: float = 50.0,
        plasticity=None,
        weight_max: float = 5.0,
        development=None,
        batch_offset: int = 0,
        batch_total: int | None = None,
    ) -> None:
        if net.n_outputs < N_MOTOR_OUTPUTS:
            raise ValueError(f"the controller needs at least {N_MOTOR_OUTPUTS} output neurons")
        self.n_actions = net.n_outputs - N_MOTOR_OUTPUTS
        self.net = net
        self.network = SpikingNetwork(
            net,
            dt_ms,
            plasticity=plasticity,
            weight_max=weight_max,
            plastic_new_only=bool(development and development.plastic_new_only),
        )
        self.development = development if self.network.develops else None
        self.difficulty = np.zeros(net.batch)
        self.woken = np.zeros(net.batch, dtype=np.int64)
        self.plasticity = plasticity if self.network.learns else None
        self._reward_baseline = np.zeros(net.batch)
        self.encoder = encoder
        self.dt_ms = dt_ms
        self.steps_per_tick = steps_per_tick
        self.bias_input = bias_input
        self.motor_threshold_hz = motor_threshold_hz
        self.action_threshold_hz = action_threshold_hz
        self.rng = rng
        # This batch may be a slice [offset, offset + B) of a larger population split
        # across processes; encoders draw randomness for the whole population and slice
        # it, so results do not depend on how the population is split.
        self.batch_offset = batch_offset
        self.batch_total = batch_total or net.batch
        b, n = net.batch, net.n_neurons
        self._i_ext = np.zeros((b, n))
        self._forced = np.zeros((steps_per_tick, b, n), dtype=bool)
        self._tick_spikes = np.zeros((b, n), dtype=np.int64)
        noninput = net.active.copy()
        noninput[:, : net.n_inputs] = False
        self._noninput = noninput
        self.n_noninput = noninput.sum(axis=1)
        self.noninput_spikes = np.zeros(b, dtype=np.int64)
        self.active_steps = np.zeros(b, dtype=np.int64)  # SNN steps while the agent lived
        self.output_spikes = np.zeros((b, net.n_outputs), dtype=np.int64)
        self.snn_steps = 0

    def replace_networks(self, net: NetworkBatch, keep: np.ndarray) -> None:
        """Swap in a rebuilt batch of networks (e.g. after births in an ecosystem) while
        keeping the running state (membrane, refractoriness, spike history, woken neurons,
        counters) of the slots in ``keep``; the other slots start fresh. Plasticity
        eligibility is reset, since the synapse list changes."""
        old = self.network
        new = SpikingNetwork(
            net,
            self.dt_ms,
            plasticity=self.plasticity if old.learns or self.plasticity else None,
            weight_max=old._wmax if old.learns else 5.0,
            plastic_new_only=bool(self.development and self.development.plastic_new_only),
        )
        keep = np.asarray(keep, dtype=bool)
        new.v[keep] = old.v[keep]
        new.ref[keep] = old.ref[keep]
        new._history[:, keep] = old._history[:, keep]
        new._ptr = old._ptr
        if new.develops and old.develops:
            new._thresh[keep] = old._thresh[keep]
            new.awake[keep] = old.awake[keep]
        if new.learns and old.learns:
            new._trace.reshape(net.batch, -1)[keep] = old._trace.reshape(net.batch, -1)[keep]
        self.net = net
        self.network = new
        fresh = ~keep
        for arr in (self.difficulty, self.woken, self._reward_baseline, self.noninput_spikes,
                    self.active_steps):  # fmt: skip
            arr[fresh] = 0
        self.output_spikes[fresh] = 0
        noninput = net.active.copy()
        noninput[:, : net.n_inputs] = False
        self._noninput = noninput
        self.n_noninput = noninput.sum(axis=1)

    def reset(self) -> None:
        self.network.reset()
        self.difficulty.fill(0.0)
        self.woken.fill(0)
        self._reward_baseline.fill(0.0)
        self.noninput_spikes.fill(0)
        self.active_steps.fill(0)
        self.output_spikes.fill(0)
        self.snn_steps = 0

    def act(
        self, observations: np.ndarray, active: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``active`` marks living agents: activity statistics only count while alive."""
        b = self.net.batch
        if active is None:
            active = np.ones(b, dtype=bool)
        x = observations
        if self.bias_input:
            x = np.concatenate([x, np.ones((b, 1))], axis=1)
        if x.shape != (b, self.net.n_inputs):
            raise ValueError(f"expected {self.net.n_inputs} input channels, got {x.shape[1]}")
        enc: EncodedInput = self.encoder.encode(
            x, self.steps_per_tick, self.rng, self.batch_offset, self.batch_total
        )
        n_in = self.net.n_inputs
        i_ext = None
        if enc.current is not None:
            self._i_ext[:, :n_in] = enc.current
            i_ext = self._i_ext
        out = self.net.output_slice
        forced_all = None
        if enc.spikes is not None:
            forced_all = self._forced  # input columns are fully overwritten every tick
            forced_all[:, :, :n_in] = enc.spikes
        tick_spikes = self._tick_spikes
        tick_spikes.fill(0)
        for t in range(self.steps_per_tick):
            forced = None if forced_all is None else forced_all[t]
            tick_spikes += self.network.step(i_ext=i_ext, forced=forced)
        counts = tick_spikes[:, out]
        noninput = np.where(self._noninput, tick_spikes, 0).sum(axis=1)
        self.noninput_spikes += noninput * active
        self.active_steps += self.steps_per_tick * active
        self.output_spikes += counts * active[:, None]
        self.snn_steps += self.steps_per_tick
        window_s = self.steps_per_tick * self.dt_ms / 1000.0
        rates = counts / window_s
        forward = self._decode(rates[:, 0] - rates[:, 1])
        turn = self._decode(rates[:, 3] - rates[:, 2])
        if self.n_actions == 0:
            return forward, turn
        return forward, turn, rates[:, N_MOTOR_OUTPUTS:] > self.action_threshold_hz

    def feedback(
        self,
        energy_delta: np.ndarray,
        alive: np.ndarray,
        stimulus_delta: np.ndarray | None = None,
    ) -> None:
        """Reward-modulated plasticity: the energy change of the last world step, compared
        with its running average (a reward prediction error), drives learning."""
        if self.development is not None:
            # Difficulty = energy taken by stimuli (damage, poison, costs of acting...).
            lost = -(stimulus_delta if stimulus_delta is not None else energy_delta)
            self.difficulty += np.maximum(lost, 0.0) * alive
            self.woken += self.network.wake(self.difficulty)
        p = self.plasticity
        if p is None:
            return
        delta = (
            stimulus_delta if p.reward == "stimuli" and stimulus_delta is not None else energy_delta
        )
        r = delta / p.reward_scale
        signal = np.clip(r - self._reward_baseline, -1.0, 1.0) * alive
        self._reward_baseline += (r - self._reward_baseline) / p.reward_baseline_steps
        self.network.apply_reward(signal)

    def energy_cost(self) -> np.ndarray | None:
        """Energy the brain uses this step: neuron_cost per awake non-input neuron."""
        if self.development is None or self.development.neuron_cost == 0:
            return None
        return self.development.neuron_cost * self.network.awake_count(self.net.n_inputs)

    def _decode(self, diff: np.ndarray) -> np.ndarray:
        thr = self.motor_threshold_hz
        return np.where(diff > thr, 1, np.where(diff < -thr, -1, 0)).astype(np.int64)

    def mean_firing_rate_hz(self) -> np.ndarray:
        """Mean firing rate of the non-input neurons while the agent was alive."""
        duration_s = np.maximum(self.active_steps, 1) * self.dt_ms / 1000.0
        return self.noninput_spikes / np.maximum(self.n_noninput, 1) / duration_s
