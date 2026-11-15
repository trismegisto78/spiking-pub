"""Grid world and simulation loop.

The world knows nothing about specific stimuli: it holds the grid geometry, the agents
and a list of objects implementing ``stimuli.base.Stimulus``. Order of one world step:

    1. every stimulus updates its own state      (Stimulus.step)
    2. observations are collected                (Stimulus.sense, concatenated in order)
    3. the controller maps observations to actions
    4. living agents turn, then move one cell; movement costs are paid
    5. every stimulus acts on the agents         (Stimulus.interact -> energy deltas)
    6. metabolic cost is paid, agents with no energy left die (if enabled)

After each step a controller that defines ``feedback(energy_delta, alive)`` receives the
energy change of every agent: the reward signal for learning during life (plasticity).

Generic actions (``network.action_outputs``) are stored in ``agents.action`` before step 5;
the core gives them no meaning, stimuli decide what they do. This is the one core extension
made for environment modification (see STATO_LAVORI.md).

In "isolated" mode every agent has its own arena (a private copy of the world, so no
competition); in "shared" mode all agents live in arena 0.

Parallel evaluation: in isolated mode a population of ``batch_total`` agents can be split
into chunks simulated by different processes; this world then holds the agents
[batch_offset, batch_offset + n_agents). Per-agent / per-arena random draws must go through
``per_arena`` so that every chunk sees exactly the numbers it would see unsplit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

    def act(self, observations: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, ...]:
        """(A, C) observations and (A,) alive mask -> (forward, turn) in {-1,0,1}, each (A,),
        optionally followed by an (A, K) boolean array of generic actions.

        Actions of dead agents are ignored by the world."""
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
        batch_offset: int = 0,
        batch_total: int | None = None,
        n_actions: int = 0,
    ) -> None:
        self.batch_offset = batch_offset
        self.batch_total = batch_total or n_agents
        if self.batch_total != n_agents and world_cfg.arenas != "isolated":
            raise ValueError("only isolated arenas can be split across processes")
        self.cfg = world_cfg
        self.agent_cfg = agent_cfg
        self.width = world_cfg.width
        self.height = world_cfg.height
        self.wrap = world_cfg.wrap
        self.stimuli = list(stimuli)
        self.n_agents = n_agents
        self.n_arenas = n_agents if world_cfg.arenas == "isolated" else 1
        arena = np.arange(n_agents) if world_cfg.arenas == "isolated" else np.zeros(n_agents)
        self.n_actions = n_actions
        self.agents = Agents.create(n_agents, arena, agent_cfg.initial_energy, n_actions)
        self.clock = clock or Clock()
        self.visited = np.zeros((n_agents, self.height, self.width), dtype=bool)

    def per_arena(self, draw: Callable[[tuple[int, ...]], np.ndarray], *tail: int) -> np.ndarray:
        """Random draw with one row per arena, chunk-invariant.

        ``draw(shape)`` is called with shape (total arenas, *tail) and the rows of the arenas
        held by this world are returned, e.g. ``world.per_arena(rng.random, count)``.
        """
        if self.n_arenas == 1 and self.batch_total == self.n_agents:
            return draw((1, *tail))
        lo = self.batch_offset
        return draw((self.batch_total, *tail))[lo : lo + self.n_arenas]

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
            lo, total = self.batch_offset, self.batch_total
            a.x[:] = rng.integers(self.width, size=total)[lo : lo + n]
            a.y[:] = rng.integers(self.height, size=total)[lo : lo + n]
            a.heading[:] = rng.integers(4, size=total)[lo : lo + n]
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
        energy_before = a.energy.copy()
        alive_before = a.alive.copy()
        for stim in self.stimuli:
            stim.step(self, t)

        out = controller.act(self.sense(), a.alive.copy())
        forward, turn = out[0], out[1]
        if self.n_actions:
            a.action[:] = (out[2] if len(out) > 2 else False) & a.alive[:, None]
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
        # Optional feedback channel: controllers that learn during life receive the energy
        # change of this step (the core does not interpret it).
        feedback = getattr(controller, "feedback", None)
        if feedback is not None:
            feedback(a.energy - energy_before, alive_before)
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
