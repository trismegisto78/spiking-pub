"""Grid world and simulation loop.

The world knows nothing about specific stimuli: it holds the grid geometry, the agents
and a list of objects implementing ``stimuli.base.Stimulus``. Order of one world step:

    1. every stimulus updates its own state      (Stimulus.step)
    2. observations are collected                (Stimulus.sense, concatenated in order)
    3. the controller maps observations to actions
    4. living agents turn, then move one cell; movement costs are paid
    5. every stimulus acts on the agents         (Stimulus.interact -> energy deltas)
    6. metabolic cost is paid, agents with no energy left die (if enabled)

In "isolated" mode every agent has its own arena (a private copy of the world, so no
competition); in "shared" mode all agents live in arena 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..config.schema import AgentConfig, WorldConfig
from ..seeding import derive_rng
from .agent import Agents, forward_vector
from .clock import Clock

if TYPE_CHECKING:
    from ..stimuli.base import Stimulus


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(A, C) observations -> (forward in {-1,0,1}, turn in {-1,0,1}), each (A,)."""
        ...


@dataclass
class EpisodeResult:
    energy_gained: np.ndarray
    final_energy: np.ndarray
    alive: np.ndarray
    steps_alive: np.ndarray
    visited_cells: np.ndarray


class World:
    def __init__(
        self,
        world_cfg: WorldConfig,
        agent_cfg: AgentConfig,
        stimuli: Sequence[Stimulus],
        n_agents: int,
        clock: Clock | None = None,
    ) -> None:
        self.cfg = world_cfg
        self.agent_cfg = agent_cfg
        self.width = world_cfg.width
        self.height = world_cfg.height
        self.wrap = world_cfg.wrap
        self.stimuli = list(stimuli)
        self.n_agents = n_agents
        self.n_arenas = n_agents if world_cfg.arenas == "isolated" else 1
        arena = np.arange(n_agents) if world_cfg.arenas == "isolated" else np.zeros(n_agents)
        self.agents = Agents.create(n_agents, arena, agent_cfg.initial_energy)
        self.clock = clock or Clock()
        self.visited = np.zeros((n_agents, self.height, self.width), dtype=bool)

    # -- channels -------------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return sum(len(s.channels) for s in self.stimuli)

    @property
    def channel_names(self) -> list[str]:
        return [f"{s.name}.{c}" for s in self.stimuli for c in s.channels]

    # -- geometry helpers usable by stimuli -------------------------------------
    def wrap_or_clip(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.wrap:
            return x % self.width, y % self.height
        return np.clip(x, 0, self.width - 1), np.clip(y, 0, self.height - 1)

    def in_bounds(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.wrap:
            return np.ones(np.broadcast(x, y).shape, dtype=bool)
        return (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)

    # -- episode ---------------------------------------------------------------
    def reset(self, seed: int, *keys: int | str) -> None:
        """Start a new episode. All randomness derives from (seed, *keys)."""
        a = self.agents
        n = self.n_agents
        rng = derive_rng(seed, *keys, "world_start")
        if self.cfg.same_start:
            a.x[:] = rng.integers(self.width)
            a.y[:] = rng.integers(self.height)
            a.heading[:] = rng.integers(4)
        else:
            a.x[:] = rng.integers(self.width, size=n)
            a.y[:] = rng.integers(self.height, size=n)
            a.heading[:] = rng.integers(4, size=n)
        a.energy[:] = self.agent_cfg.initial_energy
        a.alive[:] = True
        a.energy_gained[:] = 0.0
        a.steps_alive[:] = 0
        self.visited.fill(False)
        self._mark_visited()
        self.clock.reset()
        for i, stim in enumerate(self.stimuli):
            stim.reset(self, derive_rng(seed, *keys, "stimulus", i, stim.name))

    def sense(self) -> np.ndarray:
        parts = [np.asarray(s.sense(self.agents, self), dtype=np.float64) for s in self.stimuli]
        obs = np.concatenate(parts, axis=1) if parts else np.zeros((self.n_agents, 0))
        obs[~self.agents.alive] = 0.0
        return obs

    def step(self, controller: Controller) -> None:
        a, cfg = self.agents, self.agent_cfg
        t = self.clock.tick
        for stim in self.stimuli:
            stim.step(self, t)

        forward, turn = controller.act(self.sense())
        forward = np.where(a.alive, forward, 0)
        turn = np.where(a.alive, turn, 0)

        a.heading[:] = (a.heading + turn) % 4
        d = forward_vector(a.heading) * forward[:, None]
        nx, ny = self.wrap_or_clip(a.x + d[:, 0], a.y + d[:, 1])
        moved = (nx != a.x) | (ny != a.y)
        a.x[:], a.y[:] = nx, ny
        a.energy -= cfg.move_cost * moved + cfg.turn_cost * (turn != 0)
        self._mark_visited()

        for stim in self.stimuli:
            delta = np.asarray(stim.interact(a, self), dtype=np.float64) * a.alive
            a.energy += delta
            a.energy_gained += delta

        a.energy -= cfg.metabolic_cost * a.alive
        a.steps_alive += a.alive
        if cfg.die_at_zero:
            dead = a.energy <= 0.0
            a.alive &= ~dead
            np.maximum(a.energy, 0.0, out=a.energy)
        self.clock.advance()

    def run_episode(self, controller: Controller, seed: int, *keys: int | str) -> EpisodeResult:
        self.reset(seed, *keys)
        controller.reset()
        for _ in range(self.cfg.steps_per_episode):
            self.step(controller)
            if not self.agents.alive.any():
                break
        a = self.agents
        return EpisodeResult(
            energy_gained=a.energy_gained.copy(),
            final_energy=a.energy.copy(),
            alive=a.alive.copy(),
            steps_alive=a.steps_alive.copy(),
            visited_cells=self.visited.sum(axis=(1, 2)),
        )

    def _mark_visited(self) -> None:
        a = self.agents
        self.visited[np.arange(self.n_agents), a.y, a.x] = True
