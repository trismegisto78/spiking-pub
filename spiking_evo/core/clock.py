"""Time bookkeeping: world ticks vs. SNN integration steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Clock:
    dt_ms: float = 1.0  # length of one SNN step
    steps_per_tick: int = 10  # SNN steps per world tick
    tick: int = 0

    @property
    def tick_ms(self) -> float:
        return self.dt_ms * self.steps_per_tick

    @property
    def snn_step(self) -> int:
        return self.tick * self.steps_per_tick

    @property
    def time_ms(self) -> float:
        return self.tick * self.tick_ms

    def advance(self) -> None:
        self.tick += 1

    def reset(self) -> None:
        self.tick = 0
