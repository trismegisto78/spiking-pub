"""Others: sensing where the other living agents of the same arena are (stage E0).

The first social sense. Living agents of the same arena within ``sense_radius`` are summed
into egocentric sectors (front / left / right / back), each weighted by 1 / distance and
squashed to [0, 1] with ``1 - exp(-sense_gain * x)``, exactly like food. ``here`` is 1 when
another living agent stands on the agent's own cell. The agent never senses itself.

Only meaningful when agents share an arena (``world.arenas: groups`` or ``shared``); in
isolated arenas every channel stays 0. ``sense_gain: 0`` gives a control with the same
network inputs that never carry information. No effect on the world.

Measurement (``track_distance: true``, off by default because it costs O(agents^2) per
step): ``mate_distance`` = the agent's mean distance to its living group-mates (torus
distance on a wrapping grid), averaged over the steps it lived with at least one mate (NaN
if never). Used by the stage E1 analysis to tell a group that keeps together from one that
spreads out.
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
        track_distance: bool = False,
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
        self.track_distance = track_distance
        self._dist_sum = self._dist_n = np.zeros(0)
        self._dir_channels = tuple(c for c in self.channels if c in DIRECTIONAL_CHANNELS)
        self._kernels = sector_kernels(sense_radius, self._dir_channels)
        self._offsets = np.arange(-sense_radius, sense_radius + 1)

    def reset(self, world: World, rng: np.random.Generator) -> None:
        self._dist_sum = np.zeros(world.n_agents)
        self._dist_n = np.zeros(world.n_agents)

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
        if self.track_distance:
            self._track(agents, world)
        return np.zeros(len(agents))

    def _track(self, agents: Agents, world: World) -> None:
        alive = agents.alive
        dx = np.abs(agents.x[:, None] - agents.x[None, :]).astype(np.float64)
        dy = np.abs(agents.y[:, None] - agents.y[None, :]).astype(np.float64)
        if world.wrap:
            dx = np.minimum(dx, world.width - dx)
            dy = np.minimum(dy, world.height - dy)
        mates = (agents.arena[:, None] == agents.arena[None, :]) & alive[:, None] & alive
        np.fill_diagonal(mates, False)
        n = mates.sum(axis=1)
        has = n > 0
        total = np.where(mates, np.hypot(dx, dy), 0.0).sum(axis=1)
        self._dist_sum[has] += total[has] / n[has]
        self._dist_n[has] += 1

    def episode_stats(self) -> dict[str, np.ndarray]:
        if not self.track_distance:
            return {}
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"mate_distance": self._dist_sum / self._dist_n}
