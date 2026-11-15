"""Physical state of the agents, stored as a structure of arrays (one entry per agent).

Headings are integers 0..3 = north, east, south, west. The grid uses screen
coordinates: x grows to the east, y grows to the south.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (dx, dy) of one step forward for each heading.
HEADING_VECTORS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int64)


def forward_vector(heading: np.ndarray) -> np.ndarray:
    return HEADING_VECTORS[heading]


def right_vector(heading: np.ndarray) -> np.ndarray:
    return HEADING_VECTORS[(heading + 1) % 4]


@dataclass
class Agents:
    x: np.ndarray  # (A,) int
    y: np.ndarray  # (A,) int
    heading: np.ndarray  # (A,) int in 0..3
    energy: np.ndarray  # (A,) float
    alive: np.ndarray  # (A,) bool
    arena: np.ndarray  # (A,) int, index of the world copy the agent lives in
    energy_gained: np.ndarray  # (A,) float, sum of stimulus energy deltas while alive
    steps_alive: np.ndarray  # (A,) int
    # (A, K) bool: generic actions requested this step (K = network.action_outputs). The core
    # does not interpret them; stimuli may react to them in ``interact``.
    action: np.ndarray

    @classmethod
    def create(cls, n: int, arena: np.ndarray, initial_energy: float, n_actions: int = 0) -> Agents:
        return cls(
            x=np.zeros(n, dtype=np.int64),
            y=np.zeros(n, dtype=np.int64),
            heading=np.zeros(n, dtype=np.int64),
            energy=np.full(n, float(initial_energy)),
            alive=np.ones(n, dtype=bool),
            arena=np.asarray(arena, dtype=np.int64),
            energy_gained=np.zeros(n),
            steps_alive=np.zeros(n, dtype=np.int64),
            action=np.zeros((n, n_actions), dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.x)
