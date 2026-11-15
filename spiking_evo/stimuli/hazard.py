"""Hazard: static circular zones that drain energy from agents standing in them.

Zones are placed at the start of each episode (``count`` zones of ``radius`` cells, centres
drawn uniformly; the same layout in every arena when ``shared_layout``). An agent inside a
zone loses ``damage`` energy per world step. Sensing is the same egocentric scheme as food
(front / left / right / back sectors weighted by 1/distance, plus ``here``), on separate
channels, so the network can tell hazard from food.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .food import ALL_CHANNELS, DIRECTIONAL_CHANNELS, egocentric_sense, sector_kernels
from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World


@register("hazard")
class HazardStimulus:
    def __init__(
        self,
        count: int = 6,
        radius: float = 2.0,
        damage: float = 1.0,
        sense_radius: int = 3,
        sense_gain: float = 0.3,
        channels: list[str] | tuple[str, ...] = ("front", "left", "right", "here"),
        shared_layout: bool = True,
        name: str = "hazard",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"hazard channels must be distinct values of {ALL_CHANNELS}")
        if count < 0 or radius < 0 or damage < 0 or sense_radius < 1:
            raise ValueError("invalid hazard parameters")
        self.name = name
        self.channels = list(channels)
        self.count = count
        self.radius = radius
        self.damage = damage
        self.sense_radius = sense_radius
        self.sense_gain = sense_gain
        self.shared_layout = shared_layout
        self._dir_channels = tuple(c for c in self.channels if c in DIRECTIONAL_CHANNELS)
        self._kernels = sector_kernels(sense_radius, self._dir_channels)
        self._offsets = np.arange(-sense_radius, sense_radius + 1)

    def reset(self, world: World, rng: np.random.Generator) -> None:
        h, w, n = world.height, world.width, world.n_arenas
        if self.shared_layout:
            cy = np.broadcast_to(rng.integers(h, size=self.count), (n, self.count))
            cx = np.broadcast_to(rng.integers(w, size=self.count), (n, self.count))
        else:
            cy = world.per_arena(lambda s: rng.integers(h, size=s), self.count)
            cx = world.per_arena(lambda s: rng.integers(w, size=s), self.count)
        self.centers = np.stack([cx, cy], axis=-1)  # (arenas, count, 2) as (x, y)
        # Stamp a precomputed disk of offsets around every centre: memory is
        # O(arenas * count * disk) instead of O(arenas * count * H * W).
        r = int(np.floor(self.radius))
        oy, ox = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (ox**2 + oy**2) <= self.radius**2
        oy, ox = oy[disk], ox[disk]
        if self.shared_layout:
            # Same zones everywhere: one grid shared (read-only view) by all arenas.
            grid = np.zeros((1, h, w))
            self._stamp(grid, np.zeros(self.count, int), cy[0], cx[0], oy, ox, world)
            self.grid = np.broadcast_to(grid, (n, h, w))
        else:
            grid = np.zeros((n, h, w))
            arena = np.broadcast_to(np.arange(n)[:, None], cy.shape).reshape(-1)
            self._stamp(grid, arena, cy.reshape(-1), cx.reshape(-1), oy, ox, world)
            self.grid = grid

    @staticmethod
    def _stamp(grid, arena, cy, cx, oy, ox, world) -> None:
        ys = cy[:, None] + oy[None, :]
        xs = cx[:, None] + ox[None, :]
        if world.wrap:
            ys, xs = ys % world.height, xs % world.width
            keep = np.ones(ys.shape, dtype=bool)
        else:
            keep = (ys >= 0) & (ys < world.height) & (xs >= 0) & (xs < world.width)
        a = np.broadcast_to(arena[:, None], ys.shape)
        grid[a[keep], ys[keep], xs[keep]] = 1.0

    def step(self, world: World, t: int) -> None:
        pass  # static zones

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        return egocentric_sense(
            self.grid,
            agents,
            world,
            self.channels,
            self._dir_channels,
            self._kernels,
            self._offsets,
            self.sense_gain,
        )

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        inside = self.grid[agents.arena, agents.y, agents.x] > 0
        return np.where(inside & agents.alive, -self.damage, 0.0)

    def coverage(self) -> np.ndarray:
        """Fraction of cells covered by hazard, per arena."""
        return self.grid.mean(axis=(1, 2))
