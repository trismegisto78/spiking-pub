"""Terrain the agents can modify, with consequences they must discover.

The agents use one generic action output (``action``: its index among
``network.action_outputs``) on the cell they stand on. Every cell has a hidden soil type
that the agents cannot sense:

* fertile soil (``fertile_zones`` disks of ``fertile_radius`` cells): acting **plants a
  crop**. After ``grow_time`` steps the crop is ripe; standing on a ripe crop harvests it
  for ``harvest_energy``. Positive, but delayed.
* barren soil (everywhere else): acting **poisons the cell** for ``poison_duration`` steps
  (0 = until the end of the episode). Standing on poison costs ``poison_damage`` per step.
  Negative, and immediate.

Acting always costs ``action_cost``, and does nothing on a cell that already holds a crop
or poison. The soil itself is invisible; what the agents *can* sense are the consequences:
ripe crops (``crop_*`` channels) and poison (``poison_*`` channels), with the same
egocentric sectors as food. So an agent may learn that acting in some places pays off later,
that it can come back to harvest, and where not to act.

State is stored as event times (planting time, poison expiry), so nothing has to be
updated cell by cell at every step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .food import extract_patch, sector_kernels, sector_values
from .hazard import disk_offsets, stamp_disks
from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World

SECTORS = ("front", "left", "right", "back")
ALL_CHANNELS = tuple(f"{kind}_{s}" for kind in ("crop", "poison") for s in (*SECTORS, "here"))
NEVER = np.iinfo(np.int64).max // 4


@register("terrain")
class TerrainStimulus:
    def __init__(
        self,
        action: int = 0,
        fertile_zones: int = 10,
        fertile_radius: float = 4.0,
        grow_time: int = 100,
        harvest_energy: float = 15.0,
        poison_damage: float = 2.0,
        poison_duration: int = 200,
        action_cost: float = 0.2,
        sense_radius: int = 3,
        sense_gain: float = 1.0,
        channels: list[str] | tuple[str, ...] = (
            "crop_front",
            "crop_left",
            "crop_right",
            "crop_here",
            "poison_front",
            "poison_left",
            "poison_right",
            "poison_here",
        ),  # fmt: skip
        shared_layout: bool = True,
        name: str = "terrain",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"terrain channels must be distinct values of {ALL_CHANNELS}")
        if action < 0 or grow_time < 0 or poison_duration < 0 or sense_radius < 1:
            raise ValueError("invalid terrain parameters")
        self.name = name
        self.channels = list(channels)
        self.action = action
        self.fertile_zones = fertile_zones
        self.fertile_radius = fertile_radius
        self.grow_time = grow_time
        self.harvest_energy = harvest_energy
        self.poison_damage = poison_damage
        self.poison_duration = poison_duration
        self.action_cost = action_cost
        self.sense_radius = sense_radius
        self.sense_gain = sense_gain
        self.shared_layout = shared_layout
        self._offsets = np.arange(-sense_radius, sense_radius + 1)
        self._dirs = {
            kind: tuple(c for c in self.channels if c.startswith(kind) and not c.endswith("here"))
            for kind in ("crop", "poison")
        }
        self._kernels = {
            kind: sector_kernels(sense_radius, tuple(c.split("_")[1] for c in dirs))
            for kind, dirs in self._dirs.items()
        }
        self.t = 0

    # -- Stimulus protocol ------------------------------------------------------
    def reset(self, world: World, rng: np.random.Generator) -> None:
        if world.n_actions <= self.action:
            raise ValueError(
                f"terrain uses action {self.action}: set network.action_outputs >= "
                f"{self.action + 1}"
            )
        h, w, n = world.height, world.width, world.n_arenas
        k = self.fertile_zones
        if self.shared_layout:
            cy = np.broadcast_to(rng.integers(h, size=k), (n, k))
            cx = np.broadcast_to(rng.integers(w, size=k), (n, k))
        else:
            cy = world.per_arena(lambda s: rng.integers(h, size=s), k)
            cx = world.per_arena(lambda s: rng.integers(w, size=s), k)
        oy, ox = disk_offsets(self.fertile_radius)
        if self.shared_layout:
            soil = np.zeros((1, h, w))
            stamp_disks(soil, np.zeros(k, int), cy[0], cx[0], np.ones(k), oy, ox, world)
            self.fertile = np.broadcast_to(soil > 0, (n, h, w))
        else:
            soil = np.zeros((n, h, w))
            arena = np.broadcast_to(np.arange(n)[:, None], cy.shape).reshape(-1)
            stamp_disks(soil, arena, cy.reshape(-1), cx.reshape(-1), np.ones(n * k), oy, ox, world)
            self.fertile = soil > 0
        # Planting time + 1 (0 = no crop): 0 is also what cells outside a walled grid read.
        self.sown = np.zeros((n, h, w), dtype=np.int64)
        self.poison_until = np.zeros((n, h, w), dtype=np.int64)  # poisoned while t < value
        self.t = 0
        a = world.n_agents
        self.planted = np.zeros(a)
        self.harvested = np.zeros(a)
        self.poisoned = np.zeros(a)
        self.actions = np.zeros(a)
        self.poison_steps = np.zeros(a)

    def step(self, world: World, t: int) -> None:
        self.t = t  # state is event-based: nothing to update cell by cell

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        out = np.zeros((len(agents), len(self.channels)))
        t = self.t
        for kind, dirs in self._dirs.items():
            if dirs:
                if kind == "crop":
                    raw = extract_patch(self.sown, agents, world, self._offsets)
                    patch = (raw > 0) & (t - (raw - 1) >= self.grow_time)
                else:
                    raw = extract_patch(self.poison_until, agents, world, self._offsets)
                    patch = raw > t
                sector_values(
                    patch.astype(np.float64), agents, self._kernels[kind], self.sense_gain,
                    dirs, self.channels, out,
                )  # fmt: skip
            here = f"{kind}_here"
            if here in self.channels:
                out[:, self.channels.index(here)] = self._cell_state(agents, kind)
        return out

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        t, alive = self.t, agents.alive
        a, y, x = agents.arena, agents.y, agents.x
        delta = np.zeros(len(agents))

        # 1. Harvest ripe crops (one harvester per cell in shared arenas).
        ripe = alive & self._cell_state(agents, "crop").astype(bool)
        idx = self._one_per_cell(np.flatnonzero(ripe), agents, world)
        if idx.size:
            delta[idx] += self.harvest_energy
            self.harvested[idx] += 1
            self.sown[a[idx], y[idx], x[idx]] = 0

        # 2. Act on the current cell.
        acting = alive & agents.action[:, self.action]
        delta -= self.action_cost * acting
        self.actions += acting
        free = (self.sown[a, y, x] == 0) & (self.poison_until[a, y, x] <= t)
        idx = self._one_per_cell(np.flatnonzero(acting & free), agents, world)
        if idx.size:
            fertile = self.fertile[a[idx], y[idx], x[idx]]
            sow, spoil = idx[fertile], idx[~fertile]
            self.sown[a[sow], y[sow], x[sow]] = t + 1
            self.planted[sow] += 1
            until = NEVER if self.poison_duration == 0 else t + self.poison_duration
            self.poison_until[a[spoil], y[spoil], x[spoil]] = until
            self.poisoned[spoil] += 1

        # 3. Poison hurts whoever stands on it (including a cell poisoned just now).
        hurt = alive & (self.poison_until[a, y, x] > t)
        delta -= self.poison_damage * hurt
        self.poison_steps += hurt
        return delta

    def episode_stats(self) -> dict[str, np.ndarray]:
        return {
            "actions": self.actions.copy(),
            "planted": self.planted.copy(),
            "harvested": self.harvested.copy(),
            "poisoned": self.poisoned.copy(),
            "poison_steps": self.poison_steps.copy(),
        }

    # -- helpers -----------------------------------------------------------------
    def _cell_state(self, agents: Agents, kind: str) -> np.ndarray:
        a, y, x = agents.arena, agents.y, agents.x
        if kind == "crop":
            p = self.sown[a, y, x]
            return ((p > 0) & (self.t - (p - 1) >= self.grow_time)).astype(np.float64)
        return (self.poison_until[a, y, x] > self.t).astype(np.float64)

    @staticmethod
    def _one_per_cell(idx: np.ndarray, agents: Agents, world: World) -> np.ndarray:
        """In shared arenas keep one agent per cell (the lowest index); isolated: all."""
        if idx.size == 0 or world.n_arenas == len(agents):
            return idx
        key = (agents.arena[idx] * world.height + agents.y[idx]) * world.width + agents.x[idx]
        _, first = np.unique(key, return_index=True)
        return idx[np.sort(first)]
