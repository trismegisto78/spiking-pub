"""Food: discrete items on the grid, optionally regrowing.

Sensing is egocentric. Food within ``sense_radius`` is summed into sectors relative to the
agent heading (front / left / right / back, 90 degree cones), each item weighted by
1 / distance, then squashed to [0, 1] with ``1 - exp(-sense_gain * x)``. The ``here``
channel is 1 when food lies on the agent's own cell.

Eating happens automatically when an agent stands on a cell with food: it gains
``energy_value`` per item and the items disappear. Every missing item reappears at a
random cell with probability ``regrow_rate`` per world step (0 = fixed food).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World

DIRECTIONAL_CHANNELS = ("front", "left", "right", "back")
ALL_CHANNELS = (*DIRECTIONAL_CHANNELS, "here")


def sector_kernels(radius: int, channels: tuple[str, ...]) -> np.ndarray:
    """(4 headings, len(channels), 2r+1, 2r+1) weights; index [h, c, dy + r, dx + r]."""
    from ..core.agent import HEADING_VECTORS

    k = 2 * radius + 1
    kernels = np.zeros((4, len(channels), k, k))
    for h in range(4):
        fx, fy = HEADING_VECTORS[h]
        rx, ry = HEADING_VECTORS[(h + 1) % 4]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist = float(np.hypot(dx, dy))
                if dist == 0.0 or dist > radius:
                    continue
                f = dx * fx + dy * fy
                lat = dx * rx + dy * ry
                if f > 0 and abs(lat) <= f:
                    sector = "front"
                elif f < 0 and abs(lat) <= -f:
                    sector = "back"
                elif lat > 0:
                    sector = "right"
                else:
                    sector = "left"
                if sector in channels:
                    kernels[h, channels.index(sector), dy + radius, dx + radius] = 1.0 / dist
    return kernels


def egocentric_sense(
    grid: np.ndarray,
    agents: Agents,
    world: World,
    channels: list[str],
    dir_channels: tuple[str, ...],
    kernels: np.ndarray,
    offsets: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Shared sensing for grid-based stimuli: directional sectors + ``here``.

    ``grid`` is (arenas, H, W) with intensities (counts or 0/1).
    """
    out = np.zeros((len(agents), len(channels)))
    if dir_channels:
        ys = agents.y[:, None] + offsets
        xs = agents.x[:, None] + offsets
        valid = world.in_bounds(xs[:, None, :], ys[:, :, None])
        if world.wrap:
            ys, xs = ys % world.height, xs % world.width
        else:
            ys, xs = np.clip(ys, 0, world.height - 1), np.clip(xs, 0, world.width - 1)
        patch = grid[agents.arena[:, None, None], ys[:, :, None], xs[:, None, :]] * valid
        raw = np.einsum("aij,acij->ac", patch, kernels[agents.heading])
        vals = 1.0 - np.exp(-gain * raw)
        for j, c in enumerate(dir_channels):
            out[:, channels.index(c)] = vals[:, j]
    if "here" in channels:
        out[:, channels.index("here")] = grid[agents.arena, agents.y, agents.x] > 0
    return out


@register("food")
class FoodStimulus:
    def __init__(
        self,
        count: int = 60,
        regrow_rate: float = 0.0,
        energy_value: float = 5.0,
        sense_radius: int = 3,
        sense_gain: float = 1.0,
        channels: list[str] | tuple[str, ...] = ("front", "left", "right", "here"),
        shared_layout: bool = True,
        name: str = "food",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"food channels must be distinct values of {ALL_CHANNELS}")
        if count < 0 or not 0.0 <= regrow_rate <= 1.0 or sense_radius < 1:
            raise ValueError("invalid food parameters")
        self.name = name
        self.channels = list(channels)
        self.count = count
        self.regrow_rate = regrow_rate
        self.energy_value = energy_value
        self.sense_radius = sense_radius
        self.sense_gain = sense_gain
        self.shared_layout = shared_layout
        self._dir_channels = tuple(c for c in self.channels if c in DIRECTIONAL_CHANNELS)
        self._kernels = sector_kernels(sense_radius, self._dir_channels)
        self._offsets = np.arange(-sense_radius, sense_radius + 1)

    # -- Stimulus protocol ------------------------------------------------------
    def reset(self, world: World, rng: np.random.Generator) -> None:
        self.rng = rng
        h, w, n = world.height, world.width, world.n_arenas
        if self.count > h * w:
            raise ValueError("more food items than grid cells")
        if self.shared_layout:
            cells = np.broadcast_to(rng.choice(h * w, self.count, replace=False), (n, self.count))
        else:
            cells = np.argsort(world.per_arena(rng.random, h * w), axis=1)[:, : self.count]
        self.item_y, self.item_x = np.divmod(np.array(cells, dtype=np.int64), w)
        self.present = np.ones((n, self.count), dtype=bool)
        self._arena_of_item = np.broadcast_to(np.arange(n)[:, None], (n, self.count))
        self.grid = np.zeros((n, h, w), dtype=np.int64)
        np.add.at(self.grid, (self._arena_of_item, self.item_y, self.item_x), 1)

    def step(self, world: World, t: int) -> None:
        if self.regrow_rate <= 0.0 or self.count == 0:
            return
        # Fixed number of draws per step and arena, so an arena's food depends neither on the
        # other arenas nor on how the population is split across processes.
        u = world.per_arena(self.rng.random, self.count)
        regrow = (u < self.regrow_rate) & ~self.present
        hw = world.height * world.width
        cells = world.per_arena(lambda shape: self.rng.integers(hw, size=shape), self.count)
        if not regrow.any():
            return
        ny, nx = np.divmod(cells[regrow], world.width)
        self.item_y[regrow], self.item_x[regrow] = ny, nx
        self.present[regrow] = True
        np.add.at(self.grid, (self._arena_of_item[regrow], ny, nx), 1)

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
        delta = np.zeros(len(agents))
        amount = self.grid[agents.arena, agents.y, agents.x]
        eaters = np.flatnonzero((amount > 0) & agents.alive)
        if eaters.size == 0:
            return delta
        if world.n_arenas < len(agents):
            # Shared arenas: when several agents stand on the same food, a random one eats.
            eaters = self.rng.permutation(eaters)
            key = (agents.arena[eaters] * world.height + agents.y[eaters]) * world.width
            _, first = np.unique(key + agents.x[eaters], return_index=True)
            eaters = eaters[first]
        delta[eaters] = amount[eaters] * self.energy_value
        self.grid[agents.arena[eaters], agents.y[eaters], agents.x[eaters]] = 0
        # The items lying on the eaters' cells are gone. Only the eaters' arenas are checked,
        # so the cost does not grow with the size of the grid.
        ar = agents.arena[eaters]
        on_cell = (self.item_x[ar] == agents.x[eaters, None]) & (
            self.item_y[ar] == agents.y[eaters, None]
        )
        np.logical_and.at(self.present, ar, ~on_cell)
        return delta

    # -- helpers -----------------------------------------------------------------
    def remaining(self) -> np.ndarray:
        return self.present.sum(axis=1)
