"""Others: sensing where the other living agents of the same arena are (stage E0).

The first social sense. Living agents of the same arena within ``sense_radius`` are summed
into egocentric sectors (front / left / right / back), each weighted by 1 / distance and
squashed to [0, 1] with ``1 - exp(-sense_gain * x)``, exactly like food. ``here`` is 1 when
another living agent stands on the agent's own cell. The agent never senses itself.

Only meaningful when agents share an arena (``world.arenas: groups`` or ``shared``); in
isolated arenas every channel stays 0. ``sense_gain: 0`` gives a control with the same
network inputs that never carry information. No effect on the world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .food import DIRECTIONAL_CHANNELS, extract_patch, sector_kernels, sector_values
from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World

ALL_CHANNELS = (*DIRECTIONAL_CHANNELS, "here")


@register("others")
class OthersStimulus:
    def __init__(
        self,
        sense_radius: int = 5,
        sense_gain: float = 1.0,
        channels: list[str] | tuple[str, ...] = ("front", "left", "right", "back"),
        name: str = "others",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"others channels must be distinct values of {ALL_CHANNELS}")
        if sense_radius < 1 or sense_gain < 0:
            raise ValueError("invalid others parameters")
        self.name = name
        self.channels = list(channels)
        self.sense_radius = sense_radius
        self.sense_gain = sense_gain
        self._dir_channels = tuple(c for c in self.channels if c in DIRECTIONAL_CHANNELS)
        self._kernels = sector_kernels(sense_radius, self._dir_channels)
        self._offsets = np.arange(-sense_radius, sense_radius + 1)

    def reset(self, world: World, rng: np.random.Generator) -> None:
        pass

    def step(self, world: World, t: int) -> None:
        pass

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        out = np.zeros((len(agents), len(self.channels)))
        if self.sense_gain == 0.0 or world.n_arenas == len(agents):
            return out  # silent control, or isolated arenas: nobody else around
        alive = agents.alive
        grid = np.zeros((world.n_arenas, world.height, world.width), dtype=np.int64)
        np.add.at(grid, (agents.arena[alive], agents.y[alive], agents.x[alive]), 1)
        if self._dir_channels:
            # The agent's own cell has distance 0 and weight 0 in the kernels: no self-sensing.
            patch = extract_patch(grid, agents, world, self._offsets)
            sector_values(
                patch,
                agents,
                self._kernels,
                self.sense_gain,
                self._dir_channels,
                self.channels,
                out,
            )
        if "here" in self.channels:
            others_here = grid[agents.arena, agents.y, agents.x] - alive
            out[:, self.channels.index("here")] = others_here > 0
        return out

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        return np.zeros(len(agents))
