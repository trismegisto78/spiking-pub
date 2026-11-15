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

Crop varieties: ``crops`` is a list of ``{name, grow_time, energy, weight}``. Each fertile
zone grows one variety, drawn with probability proportional to ``weight`` (so a variety with
a small weight is a rare exception). The variety is part of the hidden soil: an agent finds
out only when the crop ripens, and a ripe crop is sensed in proportion to its energy
(``energy / max energy``). Without ``crops``, one variety uses ``grow_time`` and
``harvest_energy``.

``relocate_every`` (steps, 0 = never) redraws the hidden soil during the episode: fertile
zones and their varieties move, crops on cells whose soil changed are lost. What an agent
learned about where to farm stops being true.

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

from .base import seasonal
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
    # growth_rate scales how fast crops ripen (0.5 = twice as slow); harvest_energy scales
    # the energy of every variety; poison_damage scales the poison.
    SEASONAL = ("growth_rate", "harvest_energy", "poison_damage", "action_cost")

    def __init__(
        self,
        action: int = 0,
        fertile_zones: int = 10,
        fertile_radius: float = 4.0,
        grow_time: int = 100,
        harvest_energy: float = 15.0,
        crops: list[dict] | None = None,
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
        relocate_every: int = 0,
        name: str = "terrain",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"terrain channels must be distinct values of {ALL_CHANNELS}")
        if action < 0 or grow_time < 0 or poison_duration < 0 or sense_radius < 1:
            raise ValueError("invalid terrain parameters")
        if relocate_every < 0:
            raise ValueError("relocate_every must be >= 0")
        self.relocate_every = relocate_every
        self.name = name
        self.channels = list(channels)
        self.action = action
        self.fertile_zones = fertile_zones
        self.fertile_radius = fertile_radius
        if crops is None:
            crops = [{"name": "crop", "grow_time": grow_time, "energy": harvest_energy}]
        allowed = {"name", "grow_time", "energy", "weight"}
        for i, c in enumerate(crops):
            if set(c) - allowed or "grow_time" not in c or "energy" not in c:
                raise ValueError(f"crop {i}: needs grow_time and energy; keys {sorted(allowed)}")
            if c["grow_time"] < 0 or c.get("weight", 1.0) <= 0:
                raise ValueError(f"crop {i}: grow_time must be >= 0 and weight > 0")
        self.crop_names = [str(c.get("name", f"crop{i}")) for i, c in enumerate(crops)]
        # Index 0 = barren soil; varieties are 1..V.
        self._grow = np.array([0] + [int(c["grow_time"]) for c in crops], dtype=np.int64)
        self._energy = np.array([0.0] + [float(c["energy"]) for c in crops])
        w = np.array([float(c.get("weight", 1.0)) for c in crops])
        self._cum_weights = np.cumsum(w / w.sum())
        self._max_energy = max(float(self._energy.max()), 1e-9)
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
        self.rng = rng
        self._layout(world, rng)
        h, w, n = world.height, world.width, world.n_arenas
        # Planting time + 1 (0 = no crop): 0 is also what cells outside a walled grid read.
        # Crop growth is measured on a "growth clock" g that advances by the seasonal
        # growth rate each step (g == t without seasons). sown = g at planting + 1.
        self.sown = np.zeros((n, h, w))
        self.g = 0.0
        self._rate = 1.0
        self.poison_until = np.zeros((n, h, w), dtype=np.int64)  # poisoned while t < value
        self.t = 0
        a = world.n_agents
        self.planted = np.zeros(a)
        self.harvested = np.zeros(a)
        self.harvest_energy_total = np.zeros(a)
        self.harvested_by_variety = np.zeros((a, len(self.crop_names)))
        self.poisoned = np.zeros(a)
        self.actions = np.zeros(a)
        self.poison_steps = np.zeros(a)

    def _layout(self, world: World, rng: np.random.Generator) -> None:
        """Draw the hidden soil: fertile zones and their crop varieties."""
        h, w, n = world.height, world.width, world.n_arenas
        k = self.fertile_zones
        if self.shared_layout:
            cy = np.broadcast_to(rng.integers(h, size=k), (n, k))
            cx = np.broadcast_to(rng.integers(w, size=k), (n, k))
        else:
            cy = world.per_arena(lambda s: rng.integers(h, size=s), k)
            cx = world.per_arena(lambda s: rng.integers(w, size=s), k)
        # Variety of every zone (no extra random draw when there is only one).
        nv = len(self.crop_names)
        if nv == 1:
            var = np.ones(cy.shape)
        else:
            if self.shared_layout:
                u = np.broadcast_to(rng.random(k), cy.shape)
            else:
                u = world.per_arena(rng.random, k)
            var = np.searchsorted(self._cum_weights, u, side="right").clip(max=nv - 1) + 1.0
        oy, ox = disk_offsets(self.fertile_radius)
        if self.shared_layout:
            soil = np.zeros((1, h, w))
            stamp_disks(soil, np.zeros(k, int), cy[0], cx[0], var[0], oy, ox, world)
            self.variety = np.broadcast_to(soil.astype(np.int64), (n, h, w))
        else:
            soil = np.zeros((n, h, w))
            arena = np.broadcast_to(np.arange(n)[:, None], cy.shape).reshape(-1)
            stamp_disks(soil, arena, cy.reshape(-1), cx.reshape(-1), var.reshape(-1), oy, ox, world)
            self.variety = soil.astype(np.int64)
        self.fertile = self.variety > 0

    def step(self, world: World, t: int) -> None:
        if self.relocate_every and t > 0 and t % self.relocate_every == 0:
            old = self.variety
            self._layout(world, self.rng)
            self.sown = np.where(self.variety == old, self.sown, 0.0)  # crops on changed soil die
        if t > 0:
            self.g += self._rate
        self._rate = seasonal(self, "growth_rate")
        self.t = t  # state is event-based: nothing to update cell by cell

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        out = np.zeros((len(agents), len(self.channels)))
        t, g = self.t, self.g
        for kind, dirs in self._dirs.items():
            if dirs:
                if kind == "crop":
                    raw = extract_patch(self.sown, agents, world, self._offsets)
                    var = extract_patch(self.variety, agents, world, self._offsets)
                    ripe = (raw > 0) & (g - (raw - 1) >= self._grow[var])
                    patch = ripe * (self._energy[var] / self._max_energy)
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
        ripe = alive & (self._cell_state(agents, "crop") > 0)
        idx = self._one_per_cell(np.flatnonzero(ripe), agents, world)
        if idx.size:
            var = self.variety[a[idx], y[idx], x[idx]]
            gain = self._energy[var] * seasonal(self, "harvest_energy")
            delta[idx] += gain
            self.harvested[idx] += 1
            self.harvest_energy_total[idx] += gain
            np.add.at(self.harvested_by_variety, (idx, var - 1), 1)
            self.sown[a[idx], y[idx], x[idx]] = 0

        # 2. Act on the current cell.
        acting = alive & agents.action[:, self.action]
        delta -= self.action_cost * seasonal(self, "action_cost") * acting
        self.actions += acting
        free = (self.sown[a, y, x] == 0) & (self.poison_until[a, y, x] <= t)
        idx = self._one_per_cell(np.flatnonzero(acting & free), agents, world)
        if idx.size:
            fertile = self.fertile[a[idx], y[idx], x[idx]]
            sow, spoil = idx[fertile], idx[~fertile]
            self.sown[a[sow], y[sow], x[sow]] = self.g + 1
            self.planted[sow] += 1
            until = NEVER if self.poison_duration == 0 else t + self.poison_duration
            self.poison_until[a[spoil], y[spoil], x[spoil]] = until
            self.poisoned[spoil] += 1

        # 3. Poison hurts whoever stands on it (including a cell poisoned just now).
        hurt = alive & (self.poison_until[a, y, x] > t)
        delta -= self.poison_damage * seasonal(self, "poison_damage") * hurt
        self.poison_steps += hurt
        return delta

    def episode_stats(self) -> dict[str, np.ndarray]:
        return {
            "actions": self.actions.copy(),
            "planted": self.planted.copy(),
            "harvested": self.harvested.copy(),
            "harvest_energy": self.harvest_energy_total.copy(),
            **(
                {
                    f"harvested_{name}": self.harvested_by_variety[:, i].copy()
                    for i, name in enumerate(self.crop_names)
                }
                if len(self.crop_names) > 1
                else {}
            ),
            "poisoned": self.poisoned.copy(),
            "poison_steps": self.poison_steps.copy(),
        }

    # -- helpers -----------------------------------------------------------------
    def _cell_state(self, agents: Agents, kind: str) -> np.ndarray:
        a, y, x = agents.arena, agents.y, agents.x
        if kind == "crop":
            p = self.sown[a, y, x]
            var = self.variety[a, y, x]
            ripe = (p > 0) & (self.g - (p - 1) >= self._grow[var])
            return ripe * (self._energy[var] / self._max_energy)
        return (self.poison_until[a, y, x] > self.t).astype(np.float64)

    @staticmethod
    def _one_per_cell(idx: np.ndarray, agents: Agents, world: World) -> np.ndarray:
        """In shared arenas keep one agent per cell (the lowest index); isolated: all."""
        if idx.size == 0 or world.n_arenas == len(agents):
            return idx
        key = (agents.arena[idx] * world.height + agents.y[idx]) * world.width + agents.x[idx]
        _, first = np.unique(key, return_index=True)
        return idx[np.sort(first)]
