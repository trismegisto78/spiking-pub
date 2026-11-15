"""Hazard: static circular zones that drain energy from agents standing in them.

Zones are placed at the start of each episode (``count`` zones of ``radius`` cells, centres
drawn uniformly; the same layout in every arena when ``shared_layout``). An agent inside a
zone loses that zone's damage per world step.

``damage`` is a number (all zones alike) or a list of levels, e.g. ``[1, 3, 6]``: each zone
gets one level at random, and where zones overlap the strongest applies. Sensing is the same
egocentric scheme as food on separate channels, scaled by danger: a cell contributes
``damage / max(levels)``, so a stronger zone "smells" stronger; ``here`` is the scaled
damage of the agent's own cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import seasonal
from .food import ALL_CHANNELS, DIRECTIONAL_CHANNELS, egocentric_sense, sector_kernels
from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World


def stamp_disks(grid, arena, cy, cx, value, oy, ox, world) -> None:
    """Write ``value`` on the disk of cells (offsets oy, ox) around each centre; max where
    disks overlap. ``arena``, ``cy``, ``cx``, ``value`` have one entry per disk."""
    ys = cy[:, None] + oy[None, :]
    xs = cx[:, None] + ox[None, :]
    if world.wrap:
        ys, xs = ys % world.height, xs % world.width
        keep = np.ones(ys.shape, dtype=bool)
    else:
        keep = (ys >= 0) & (ys < world.height) & (xs >= 0) & (xs < world.width)
    a = np.broadcast_to(arena[:, None], ys.shape)
    d = np.broadcast_to(np.asarray(value, dtype=np.float64)[:, None], ys.shape)
    np.maximum.at(grid, (a[keep], ys[keep], xs[keep]), d[keep])


def disk_offsets(radius: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(np.floor(radius))
    oy, ox = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (ox**2 + oy**2) <= radius**2
    return oy[disk], ox[disk]


@register("hazard")
class HazardStimulus:
    SEASONAL = ("damage",)

    def __init__(
        self,
        count: int = 6,
        radius: float = 2.0,
        damage: float | list[float] = 1.0,
        sense_radius: int = 3,
        sense_gain: float = 0.3,
        channels: list[str] | tuple[str, ...] = ("front", "left", "right", "here"),
        shared_layout: bool = True,
        name: str = "hazard",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"hazard channels must be distinct values of {ALL_CHANNELS}")
        levels = [float(d) for d in (damage if isinstance(damage, list | tuple) else [damage])]
        if count < 0 or radius < 0 or not levels or min(levels) < 0 or sense_radius < 1:
            raise ValueError("invalid hazard parameters")
        self.levels = np.array(levels)
        self.max_damage = max(float(self.levels.max()), 1e-9)
        self.name = name
        self.channels = list(channels)
        self.count = count
        self.radius = radius
        self.sense_radius = sense_radius
        self.sense_gain = sense_gain
        self.shared_layout = shared_layout
        self._dir_channels = tuple(c for c in self.channels if c in DIRECTIONAL_CHANNELS)
        self._kernels = sector_kernels(sense_radius, self._dir_channels)
        self._offsets = np.arange(-sense_radius, sense_radius + 1)

    def reset(self, world: World, rng: np.random.Generator) -> None:
        self.steps_inside = np.zeros(world.n_agents, dtype=np.int64)
        self.damage_taken = np.zeros(world.n_agents)
        h, w, n = world.height, world.width, world.n_arenas
        if self.shared_layout:
            cy = np.broadcast_to(rng.integers(h, size=self.count), (n, self.count))
            cx = np.broadcast_to(rng.integers(w, size=self.count), (n, self.count))
        else:
            cy = world.per_arena(lambda s: rng.integers(h, size=s), self.count)
            cx = world.per_arena(lambda s: rng.integers(w, size=s), self.count)
        self.centers = np.stack([cx, cy], axis=-1)  # (arenas, count, 2) as (x, y)
        # Damage level of every zone (no extra random draw when there is a single level).
        nl = len(self.levels)
        if nl == 1:
            lv = np.zeros(cy.shape, dtype=np.int64)
        elif self.shared_layout:
            lv = np.broadcast_to(rng.integers(nl, size=self.count), cy.shape)
        else:
            lv = world.per_arena(lambda s: rng.integers(nl, size=s), self.count)
        self.zone_damage = self.levels[lv]  # (arenas, count)
        # Stamp a precomputed disk of offsets around every centre: memory is
        # O(arenas * count * disk) instead of O(arenas * count * H * W).
        oy, ox = disk_offsets(self.radius)
        if self.shared_layout:
            # Same zones everywhere: one grid shared (read-only view) by all arenas.
            grid = np.zeros((1, h, w))
            stamp_disks(
                grid, np.zeros(self.count, int), cy[0], cx[0], self.zone_damage[0], oy, ox, world
            )
            self.grid = np.broadcast_to(grid, (n, h, w))
            self._sense_grid = np.broadcast_to(grid / self.max_damage, (n, h, w))
        else:
            grid = np.zeros((n, h, w))
            arena = np.broadcast_to(np.arange(n)[:, None], cy.shape).reshape(-1)
            stamp_disks(
                grid,
                arena,
                cy.reshape(-1),
                cx.reshape(-1),
                self.zone_damage.reshape(-1),
                oy,
                ox,
                world,
            )
            self.grid = grid
            self._sense_grid = grid / self.max_damage

    def step(self, world: World, t: int) -> None:
        pass  # static zones

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        return egocentric_sense(
            self._sense_grid,
            agents,
            world,
            self.channels,
            self._dir_channels,
            self._kernels,
            self._offsets,
            self.sense_gain,
            here_scale=1.0,
        )

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        dmg = np.where(agents.alive, self.grid[agents.arena, agents.y, agents.x], 0.0)
        dmg = dmg * seasonal(self, "damage")
        self.steps_inside += dmg > 0
        self.damage_taken += dmg
        return -dmg

    def episode_stats(self) -> dict[str, np.ndarray]:
        return {
            "steps_inside": self.steps_inside.astype(np.float64),
            "damage_taken": self.damage_taken.copy(),
        }

    def coverage(self) -> np.ndarray:
        """Fraction of cells covered by hazard, per arena."""
        return (self.grid > 0).mean(axis=(1, 2))
