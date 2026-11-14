"""Sensor intensities in [0, 1] -> input for the SNN, one world tick at a time.

Every encoder turns a (B, C) intensity array into the drive for ``n_steps`` SNN steps:
either a (n_steps, B, C) boolean array of forced spikes, or a (B, C) constant current.

``x`` may be the slice [offset, offset + B) of a population of ``total`` agents. Stochastic
encoders draw random numbers for all ``total`` agents and keep their slice, so the spikes an
agent receives do not depend on how the population is split across processes.
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

    def encode(
        self,
        x: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
        offset: int = 0,
        total: int | None = None,
    ) -> EncodedInput:
        p = np.clip(x, 0.0, 1.0) * self.p_max
        b, c = x.shape
        u = rng.random((n_steps, total or b, c))[:, offset : offset + b]
        return EncodedInput(spikes=u < p)


class LatencyEncoder:
    """Temporal coding: one spike per tick, earlier for stronger intensities.

    Intensity 1 spikes at the first step of the tick, intensities close to 0 at the last
    step, intensity 0 does not spike.
    """

    stochastic = False

    def encode(
        self,
        x: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
        offset: int = 0,
        total: int | None = None,
    ) -> EncodedInput:
        x = np.clip(x, 0.0, 1.0)
        t_spike = np.floor((1.0 - x) * (n_steps - 1) + 1e-9).astype(np.int64)
        steps = np.arange(n_steps).reshape(n_steps, *([1] * x.ndim))
        return EncodedInput(spikes=(steps == t_spike) & (x > 0.0))


class CurrentEncoder:
    """Intensity injected as a constant current into LIF input neurons."""

    stochastic = False

    def __init__(self, gain: float) -> None:
        self.gain = gain

    def encode(
        self,
        x: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
        offset: int = 0,
        total: int | None = None,
    ) -> EncodedInput:
        return EncodedInput(current=np.clip(x, 0.0, 1.0) * self.gain)


def make_encoder(kind: str, *, max_rate_hz: float, dt_ms: float, current_gain: float):
    if kind == "rate":
        return RateEncoder(max_rate_hz, dt_ms)
    if kind == "latency":
        return LatencyEncoder()
    if kind == "current":
        return CurrentEncoder(current_gain)
    raise ValueError(f"unknown encoder {kind!r}")
