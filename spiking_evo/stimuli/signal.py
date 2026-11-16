"""Signal: a channel agents may choose to emit on, heard by the others nearby (stage E1).

Emitting: a living agent whose generic action output ``action`` fires during a step emits
for that step and pays ``emit_cost`` energy. Nothing else happens: what a signal means, if
anything, is up to evolution.

Hearing: at the next step every living agent hears the emitters of its own arena within
``sense_radius`` (positions at emission time), summed into egocentric sectors (front / left
/ right / back), each weighted by 1 / distance and squashed to [0, 1] with
``1 - exp(-sense_gain * x)``, like food and the others. An agent never hears itself.
``sense_gain: 0`` is the mute control: same inputs and same costs, never any information.

Statistics (per agent, for measuring whether the signal carries information): an agent is
*fed* when its energy gained from the stimuli increased during one of the last
``fed_window`` steps (it is eating, i.e. on food). ``emitted`` counts emissions;
``steps_fed`` / ``emitted_fed`` and ``steps_unfed`` / ``emitted_unfed`` split living steps
and emissions by that state. List this stimulus after the food stimuli, so the energy they
give in the same step counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .food import DIRECTIONAL_CHANNELS, extract_patch, sector_kernels, sector_values
from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World

FAR = 1 << 30
STATS = ("emitted", "steps_fed", "emitted_fed", "steps_unfed", "emitted_unfed")


@register("signal")
class SignalStimulus:
    def __init__(
        self,
        action: int = 0,
        sense_radius: int = 6,
        sense_gain: float = 1.0,
        emit_cost: float = 0.01,
        fed_window: int = 10,
        channels: list[str] | tuple[str, ...] = DIRECTIONAL_CHANNELS,
        name: str = "signal",
    ) -> None:
        bad = [c for c in channels if c not in DIRECTIONAL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"signal channels must be distinct values of {DIRECTIONAL_CHANNELS}")
        if action < 0 or sense_radius < 1 or sense_gain < 0 or emit_cost < 0 or fed_window < 1:
            raise ValueError("invalid signal parameters")
        self.name = name
        self.channels = list(channels)
        self.action = action
        self.sense_radius = sense_radius
        self.sense_gain = sense_gain
        self.emit_cost = emit_cost
        self.fed_window = fed_window
        self._kernels = sector_kernels(sense_radius, tuple(self.channels))
        self._offsets = np.arange(-sense_radius, sense_radius + 1)

    def reset(self, world: World, rng: np.random.Generator) -> None:
        n = world.n_agents
        self._grid = np.zeros((world.n_arenas, world.height, world.width), dtype=np.int64)
        self._any = False
        self._prev_gained = world.agents.energy_gained.copy()
        self._since_fed = np.full(n, FAR, dtype=np.int64)
        self._stats = {s: np.zeros(n, dtype=np.int64) for s in STATS}

    def step(self, world: World, t: int) -> None:
        pass

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        out = np.zeros((len(agents), len(self.channels)))
        if self.sense_gain == 0.0 or not self._any:
            return out
        patch = extract_patch(self._grid, agents, world, self._offsets)
        sector_values(
            patch,
            agents,
            self._kernels,
            self.sense_gain,
            tuple(self.channels),
            self.channels,
            out,
        )
        return out

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        alive = agents.alive
        fire = np.zeros(len(agents), dtype=bool)
        if agents.action.shape[1] > self.action:
            fire = agents.action[:, self.action] & alive
        # What the others will hear at the next step.
        self._grid.fill(0)
        np.add.at(self._grid, (agents.arena[fire], agents.y[fire], agents.x[fire]), 1)
        self._any = bool(fire.any())
        # Information content: emissions while eating vs not.
        ate = agents.energy_gained > self._prev_gained
        self._prev_gained = agents.energy_gained.copy()
        self._since_fed = np.where(ate, 0, np.minimum(self._since_fed + 1, FAR))
        fed = (self._since_fed < self.fed_window) & alive
        unfed = ~fed & alive
        s = self._stats
        s["emitted"] += fire
        s["steps_fed"] += fed
        s["emitted_fed"] += fire & fed
        s["steps_unfed"] += unfed
        s["emitted_unfed"] += fire & unfed
        return -self.emit_cost * fire

    def episode_stats(self) -> dict[str, np.ndarray]:
        return {k: v.astype(np.float64) for k, v in self._stats.items()}
