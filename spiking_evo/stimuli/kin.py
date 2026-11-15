"""Kin: sensing one's own offspring and parent, and feeding juveniles.

Needs the per-agent ids and parent ids that the ecosystem publishes in ``world.extra``
(``"id"``, ``"parent"``); without them the stimulus senses nothing and nobody can feed.

Senses (egocentric sectors within ``sense_radius``, weighted 1/distance, squashed to
[0, 1]):
    child_front / child_left / child_right / child_back / child_here
        own juvenile offspring (``child_here``: one is within reach, i.e. feedable)
    parent_front / parent_left / parent_right / parent_back / parent_here
        own parent (alive)

Feeding: an adult whose action output ``action`` fires, and that has more than
``min_donor_energy``, gives ``feed_amount`` energy to the poorest juvenile within reach
(Chebyshev distance <= 1); the juvenile receives ``feed_efficiency`` of it. With
``kin_only`` (default) only one's own offspring can be fed; otherwise any juvenile.
Whether to feed, and whom, is up to the evolved brain: nothing forces it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.agent import HEADING_VECTORS
from .registry import register

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World

SECTORS = ("front", "left", "right", "back", "here")
ALL_CHANNELS = tuple(f"{who}_{s}" for who in ("child", "parent") for s in SECTORS)


@register("kin")
class KinStimulus:
    def __init__(
        self,
        action: int = 0,
        sense_radius: int = 5,
        sense_gain: float = 1.0,
        feed_amount: float = 2.0,
        feed_efficiency: float = 0.8,
        min_donor_energy: float = 20.0,
        kin_only: bool = True,
        channels: list[str] | tuple[str, ...] = (
            "child_front",
            "child_left",
            "child_right",
            "child_here",
            "parent_front",
            "parent_left",
            "parent_right",
            "parent_here",
        ),  # fmt: skip
        name: str = "kin",
    ) -> None:
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad or len(set(channels)) != len(channels):
            raise ValueError(f"kin channels must be distinct values of {ALL_CHANNELS}")
        if not 0 <= feed_efficiency <= 1 or feed_amount < 0 or sense_radius < 1:
            raise ValueError("invalid kin parameters")
        self.name = name
        self.channels = list(channels)
        self.action = action
        self.sense_radius = sense_radius
        self.sense_gain = sense_gain
        self.feed_amount = feed_amount
        self.feed_efficiency = feed_efficiency
        self.min_donor_energy = min_donor_energy
        self.kin_only = kin_only

    # -- Stimulus protocol ------------------------------------------------------
    def reset(self, world: World, rng: np.random.Generator) -> None:
        n = world.n_agents
        self.feed_events = np.zeros(n)
        self.energy_given = np.zeros(n)
        self.energy_received = np.zeros(n)

    def step(self, world: World, t: int) -> None:
        pass

    def _offsets(self, agents: Agents, world: World, idx: np.ndarray):
        x, y = agents.x[idx], agents.y[idx]
        dx = x[None, :] - x[:, None]  # (observer, target), living agents only
        dy = y[None, :] - y[:, None]
        if world.wrap:
            dx = (dx + world.width // 2) % world.width - world.width // 2
            dy = (dy + world.height // 2) % world.height - world.height // 2
        return dx, dy

    def _relations(self, agents: Agents, world: World, idx: np.ndarray):
        ids, parent = world.extra.get("id"), world.extra.get("parent")
        if ids is None or parent is None:
            return None
        ids, parent, stage = ids[idx], parent[idx], agents.stage[idx]
        child = (parent[None, :] == ids[:, None]) & (stage[None, :] == 1)
        par = ids[None, :] == parent[:, None]
        np.fill_diagonal(child, False)
        np.fill_diagonal(par, False)
        return child, par

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        out = np.zeros((len(agents), len(self.channels)))
        idx = np.flatnonzero(agents.alive)
        rel = self._relations(agents, world, idx)
        if rel is None or idx.size < 2 or not (rel[0].any() or rel[1].any()):
            return out
        dx, dy = self._offsets(agents, world, idx)
        dist = np.hypot(dx, dy)
        heading = agents.heading[idx]
        fwd = HEADING_VECTORS[heading]
        rgt = HEADING_VECTORS[(heading + 1) % 4]
        f = dx * fwd[:, 0:1] + dy * fwd[:, 1:2]
        lat = dx * rgt[:, 0:1] + dy * rgt[:, 1:2]
        near = (dist <= self.sense_radius) & (dist > 0)
        here = np.maximum(np.abs(dx), np.abs(dy)) <= 1
        sectors = {
            "front": (f > 0) & (np.abs(lat) <= f),
            "back": (f < 0) & (np.abs(lat) <= -f),
            "right": (lat > 0) & (np.abs(lat) > np.abs(f)),
            "left": (lat < 0) & (np.abs(lat) > np.abs(f)),
        }
        w = np.where(near, 1.0 / np.maximum(dist, 1.0), 0.0)
        for who, mask in zip(("child", "parent"), rel, strict=True):
            for s in SECTORS:
                c = f"{who}_{s}"
                if c not in self.channels:
                    continue
                if s == "here":
                    v = (mask & here).any(axis=1).astype(np.float64)
                else:
                    v = 1.0 - np.exp(-self.sense_gain * (w * (mask & sectors[s])).sum(axis=1))
                out[idx, self.channels.index(c)] = v
        return out

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        delta = np.zeros(len(agents))
        if agents.action.shape[1] <= self.action:
            return delta
        donor_mask = (
            agents.alive
            & agents.action[:, self.action]
            & (agents.stage == 0)
            & (agents.energy > self.min_donor_energy)
        )
        if not donor_mask.any():
            return delta
        idx = np.flatnonzero(agents.alive)
        rel = self._relations(agents, world, idx)
        if rel is None:
            return delta
        child, _ = rel
        dx, dy = self._offsets(agents, world, idx)
        reach = np.maximum(np.abs(dx), np.abs(dy)) <= 1
        juvenile = agents.stage[idx] == 1
        for k in np.flatnonzero(donor_mask[idx]):
            d = idx[k]
            targets = (child[k] if self.kin_only else juvenile.copy()) & reach[k]
            targets[k] = False
            cand = idx[targets]
            if cand.size == 0:
                continue
            j = cand[np.argmin(agents.energy[cand] + delta[cand])]
            delta[d] -= self.feed_amount
            delta[j] += self.feed_amount * self.feed_efficiency
            self.feed_events[d] += 1
            self.energy_given[d] += self.feed_amount
            self.energy_received[j] += self.feed_amount * self.feed_efficiency
        return delta

    def episode_stats(self) -> dict[str, np.ndarray]:
        return {
            "feed_events": self.feed_events.copy(),
            "energy_given": self.energy_given.copy(),
            "energy_received": self.energy_received.copy(),
        }
