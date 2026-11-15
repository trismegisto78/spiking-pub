"""Seasons: a cycle that changes the other stimuli during an agent's life.

``seasons`` is a list of ``{name, length, modifiers, living_cost}``; the cycle repeats every
``sum(length)`` world steps. ``modifiers`` multiplies parameters of the other stimuli,
addressed as ``"<stimulus name>.<parameter>"``, for example::

    seasons:
      - { name: spring, length: 125, modifiers: { food.regrow_rate: 1.5 } }
      - { name: summer, length: 125, modifiers: { hazard.damage: 2.0 } }
      - { name: autumn, length: 125, modifiers: { terrain.harvest_energy: 1.5 } }
      - { name: winter, length: 125, modifiers: { food.regrow_rate: 0.0,
                                                  terrain.growth_rate: 0.25 } }

``living_cost`` (default 0) is an extra energy cost per step for every living agent during
that season, on top of ``agent.metabolic_cost``: a harsh winter makes simply staying alive
expensive. It is charged by this stimulus, so it appears in the agents' energy balance (and
in ``energy_<season>``) like any other stimulus effect.

Each stimulus lists the parameters it accepts in ``SEASONAL`` (food: regrow_rate,
energy_value; hazard: damage; terrain: growth_rate, harvest_energy, poison_damage). A
parameter not named in the current season has multiplier 1.

Agents can perceive the time of year through ``phase_sin`` / ``phase_cos`` (a smooth clock,
like daylight or temperature) and ``<season name>`` one-hot channels; ``channels: []``
hides the seasons completely. With ``random_phase`` every episode starts at a random point of
the cycle (the same for all arenas), so agents cannot rely on the season at birth.

List ``season`` first among the stimuli: it updates the multipliers in its ``step``, and
stimuli listed before it see the change one step late.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World


@register("season")
class SeasonStimulus:
    def __init__(
        self,
        seasons: list[dict] | None = None,
        channels: list[str] | tuple[str, ...] = ("phase_sin", "phase_cos"),
        random_phase: bool = False,
        name: str = "season",
    ) -> None:
        if not seasons:
            raise ValueError("season: give at least one season")
        allowed = {"name", "length", "modifiers", "living_cost"}
        for i, s in enumerate(seasons):
            if set(s) - allowed or "length" not in s or int(s["length"]) < 1:
                raise ValueError(f"season {i}: needs length >= 1; keys {sorted(allowed)}")
            if float(s.get("living_cost", 0.0)) < 0:
                raise ValueError(f"season {i}: living_cost must be >= 0")
            for key, value in (s.get("modifiers") or {}).items():
                if key.count(".") != 1 or float(value) < 0:
                    raise ValueError(f"season {i}: modifier {key!r} must be 'stimulus.param' >= 0")
        self.season_names = [str(s.get("name", f"season{i}")) for i, s in enumerate(seasons)]
        valid_channels = ("phase_sin", "phase_cos", *self.season_names)
        bad = [c for c in channels if c not in valid_channels]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"season channels must be distinct values of {valid_channels}")
        self.name = name
        self.channels = list(channels)
        self.seasons = seasons
        self.random_phase = random_phase
        self.lengths = np.array([int(s["length"]) for s in seasons])
        self.living_costs = np.array([float(s.get("living_cost", 0.0)) for s in seasons])
        self.period = int(self.lengths.sum())
        self._starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.offset = 0
        self.current = 0
        self._targets: dict[str, object] = {}

    # -- Stimulus protocol ------------------------------------------------------
    def reset(self, world: World, rng: np.random.Generator) -> None:
        by_name = {s.name: s for s in world.stimuli if s is not self}
        self._targets = {}
        for s in self.seasons:
            for key in s.get("modifiers") or {}:
                target, param = key.split(".")
                if target not in by_name:
                    raise ValueError(f"season modifier {key!r}: no stimulus named {target!r}")
                accepted = getattr(by_name[target], "SEASONAL", ())
                if param not in accepted:
                    raise ValueError(f"season modifier {key!r}: {target} accepts {list(accepted)}")
                self._targets[target] = by_name[target]
        self.offset = int(rng.integers(self.period)) if self.random_phase else 0
        a, k = world.n_agents, len(self.seasons)
        self.energy_by_season = np.zeros((a, k))
        self.actions_by_season = np.zeros((a, k))
        self.steps_by_season = np.zeros((a, k))
        self._last_gained = np.zeros(a)
        self._apply(0)

    def step(self, world: World, t: int) -> None:
        # Book what happened during the previous step to the season it belonged to.
        if t > 0:
            ag = world.agents
            prev = self.current
            self.energy_by_season[:, prev] += ag.energy_gained - self._last_gained
            self._last_gained = ag.energy_gained.copy()
            self.steps_by_season[:, prev] += ag.alive
            if ag.action.shape[1]:
                self.actions_by_season[:, prev] += ag.action.any(axis=1)
        self._apply(t)

    def episode_stats(self) -> dict[str, np.ndarray]:
        """Energy gained, actions and living steps per season (the last step of an episode
        is not booked)."""
        out = {}
        for i, name in enumerate(self.season_names):
            out[f"energy_{name}"] = self.energy_by_season[:, i].copy()
            out[f"actions_{name}"] = self.actions_by_season[:, i].copy()
            out[f"steps_{name}"] = self.steps_by_season[:, i].copy()
        return out

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        out = np.zeros((len(agents), len(self.channels)))
        angle = 2 * np.pi * self._phase / self.period
        values = {"phase_sin": 0.5 + 0.5 * np.sin(angle), "phase_cos": 0.5 + 0.5 * np.cos(angle)}
        values[self.season_names[self.current]] = 1.0
        for j, c in enumerate(self.channels):
            out[:, j] = values.get(c, 0.0)
        return out

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        return -self.living_costs[self.current] * agents.alive

    # -- helpers -----------------------------------------------------------------
    def season_at(self, t: int) -> int:
        phase = (t + self.offset) % self.period
        return int(np.searchsorted(self._starts, phase, side="right") - 1)

    def _apply(self, t: int) -> None:
        self._phase = (t + self.offset) % self.period
        self.current = self.season_at(t)
        mods = self.seasons[self.current].get("modifiers") or {}
        for target_name, target in self._targets.items():
            target._season = {
                key.split(".")[1]: float(v)
                for key, v in mods.items()
                if key.split(".")[0] == target_name
            }
