"""The extension point of the simulation: everything that the agents can sense or that
affects them goes through a ``Stimulus``.

Note on the signatures: the specification describes ``sense``/``interact`` per agent.
Here they receive the whole ``Agents`` batch (structure of arrays) and return one row per
agent. A per-agent Python call would cost population x steps x stimuli calls per
generation (~10^6 for the baseline), which dominates run time; batched calls keep the same
semantics while letting each stimulus vectorise with NumPy.

Reproducibility rule: random numbers that differ per arena must be drawn through
``world.per_arena(rng.<method>, ...)``. The population may be split across processes, and
this keeps every arena's random stream independent of the split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ..core.agent import Agents
    from ..core.world import World


@runtime_checkable
class Stimulus(Protocol):
    name: str
    channels: list[str]  # names of the sensory channels this stimulus occupies, in order

    def reset(self, world: World, rng: np.random.Generator) -> None:
        """Initialise the stimulus at the start of an episode. ``rng`` is private to it."""

    def step(self, world: World, t: int) -> None:
        """Update the stimulus state (regrowth, movement, ...) at world tick ``t``."""

    def sense(self, agents: Agents, world: World) -> np.ndarray:
        """(A, len(channels)) intensities in [0, 1] perceived by each agent."""

    def interact(self, agents: Agents, world: World) -> np.ndarray:
        """Apply effects on the agents; return the (A,) energy delta.

        Agents with ``agents.alive == False`` must not interact (e.g. not eat).
        """
