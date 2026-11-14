"""Sensor intensities in [0, 1] -> input for the SNN, one world tick at a time.

Every encoder turns a (B, C) intensity array into the drive for ``n_steps`` SNN steps:
either a (n_steps, B, C) boolean array of forced spikes, or a (B, C) constant current.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EncodedInput:
    spikes: np.ndarray | None = None  # (T, B, C) bool
    current: np.ndarray | None = None  # (B, C) float


class RateEncoder:
    """Bernoulli (Poisson-like) rate coding: rate = intensity * max_rate_hz."""

    stochastic = True

    def __init__(self, max_rate_hz: float, dt_ms: float) -> None:
        self.p_max = max_rate_hz * dt_ms / 1000.0
        if not 0.0 <= self.p_max <= 1.0:
            raise ValueError("max_rate_hz * dt_ms must be <= 1000 (at most one spike per step)")

    def encode(self, x: np.ndarray, n_steps: int, rng: np.random.Generator) -> EncodedInput:
        p = np.clip(x, 0.0, 1.0) * self.p_max
        return EncodedInput(spikes=rng.random((n_steps, *x.shape)) < p)


class LatencyEncoder:
    """Temporal coding: one spike per tick, earlier for stronger intensities.

    Intensity 1 spikes at the first step of the tick, intensities close to 0 at the last
    step, intensity 0 does not spike.
    """

    stochastic = False

    def encode(self, x: np.ndarray, n_steps: int, rng: np.random.Generator) -> EncodedInput:
        x = np.clip(x, 0.0, 1.0)
        t_spike = np.floor((1.0 - x) * (n_steps - 1) + 1e-9).astype(np.int64)
        steps = np.arange(n_steps).reshape(n_steps, *([1] * x.ndim))
        return EncodedInput(spikes=(steps == t_spike) & (x > 0.0))


class CurrentEncoder:
    """Intensity injected as a constant current into LIF input neurons."""

    stochastic = False

    def __init__(self, gain: float) -> None:
        self.gain = gain

    def encode(self, x: np.ndarray, n_steps: int, rng: np.random.Generator) -> EncodedInput:
        return EncodedInput(current=np.clip(x, 0.0, 1.0) * self.gain)


def make_encoder(kind: str, *, max_rate_hz: float, dt_ms: float, current_gain: float):
    if kind == "rate":
        return RateEncoder(max_rate_hz, dt_ms)
    if kind == "latency":
        return LatencyEncoder()
    if kind == "current":
        return CurrentEncoder(current_gain)
    raise ValueError(f"unknown encoder {kind!r}")
