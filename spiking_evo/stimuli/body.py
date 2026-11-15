"""Body: what an agent feels about itself.

Channels: ``energy`` (own energy / ``energy_scale``, clipped to [0, 1]), ``juvenile`` (1 while
the agent is a juvenile, see ``agents.stage``) and ``age`` (age / ``age_scale``, clipped).
No effect on the world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World

ALL_CHANNELS = ("energy", "juvenile", "age")


@register("body")
class BodyStimulus:
    def __init__(
        self,
        channels: list[str] | tuple[str, ...] = ("energy", "juvenile"),
        energy_scale: float = 100.0,
        age_scale: float = 3000.0,
        name: str = "body",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"body channels must be distinct values of {ALL_CHANNELS}")
        self.name = name
        self.channels = list(channels)
        self.energy_scale = energy_scale
        self.age_scale = age_scale

    def reset(self, world: World, rng: np.random.Generator) -> None:
        pass

    def step(self, world: World, t: int) -> None:
        pass

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        values = {
            "energy": np.clip(agents.energy / self.energy_scale, 0.0, 1.0),
            "juvenile": (agents.stage == 1).astype(np.float64),
            "age": np.clip(agents.age / self.age_scale, 0.0, 1.0),
        }
        return np.stack([values[c] for c in self.channels], axis=1)

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        return np.zeros(len(agents))
