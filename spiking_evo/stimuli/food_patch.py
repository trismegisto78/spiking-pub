"""Food patches: food that grows in clusters, optionally moving (stage E0).

Like ``food`` (same senses, same eating), but the items belong to ``patches`` clusters:
item ``i`` belongs to patch ``i % patches`` and lies at its centre plus a Gaussian offset of
standard deviation ``patch_radius`` cells (rounded; several items may share a cell). A
missing item regrows, with probability ``regrow_rate`` per step, around the *current*
centre of its patch. Every ``relocate_every`` steps (0 = never) all patch centres jump to new
random places: items already on the ground stay until eaten, new ones grow at the new
places. Moving patches make where-the-food-is information that ages, so finding it again,
alone or by following the others, has a value.

With ``shared_layout`` every arena gets the same centres; otherwise each arena draws its
own (through ``world.per_arena``, so results do not depend on how the population is split).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import seasonal
from .food import FoodStimulus
from .registry import register

if TYPE_CHECKING:
    from ..core.world import World


@register("food_patch")
class FoodPatchStimulus(FoodStimulus):
    def __init__(
        self,
        count: int = 60,
        patches: int = 4,
        patch_radius: float = 2.0,
        relocate_every: int = 0,
        regrow_rate: float = 0.0,
        energy_value: float = 5.0,
        sense_radius: int = 3,
        sense_gain: float = 1.0,
        channels: list[str] | tuple[str, ...] = ("front", "left", "right", "here"),
        shared_layout: bool = True,
        name: str = "food",
    ) -> None:
        super().__init__(
            count,
            regrow_rate,
            energy_value,
            sense_radius,
            sense_gain,
            channels,
            shared_layout,
            name,
        )
        if patches < 1 or patch_radius < 0 or relocate_every < 0:
            raise ValueError("invalid food_patch parameters")
        self.patches = patches
        self.patch_radius = patch_radius
        self.relocate_every = relocate_every
        self._patch_of_item = np.arange(count) % patches

    # -- placement ----------------------------------------------------------------------
    def _draw(self, world: World, draw, *tail: int) -> np.ndarray:
        """(arenas, *tail) draw: identical in every arena with ``shared_layout``."""
        if self.shared_layout:
            return np.array(np.broadcast_to(draw(tail), (world.n_arenas, *tail)))
        return world.per_arena(draw, *tail)

    def _new_centers(self, world: World) -> np.ndarray:
        cx = self._draw(world, lambda s: self.rng.integers(world.width, size=s), self.patches)
        cy = self._draw(world, lambda s: self.rng.integers(world.height, size=s), self.patches)
        return np.stack([cx, cy], axis=-1)  # (arenas, patches, 2) as (x, y)

    def _around_centers(self, world: World, offsets: np.ndarray):
        """Cells of every item placed around its patch centre; offsets (arenas, count, 2)."""
        c = self.centers[:, self._patch_of_item]  # (arenas, count, 2)
        pos = np.rint(c + self.patch_radius * offsets).astype(np.int64)
        return world.wrap_or_clip(pos[..., 0], pos[..., 1])

    # -- Stimulus protocol ----------------------------------------------------------------
    def reset(self, world: World, rng: np.random.Generator) -> None:
        self.rng = rng
        n, h, w = world.n_arenas, world.height, world.width
        self.eaten = np.zeros(world.n_agents, dtype=np.int64)
        self.centers = self._new_centers(world)
        offsets = self._draw(world, lambda s: rng.normal(size=s), self.count, 2)
        self.item_x, self.item_y = self._around_centers(world, offsets)
        self.present = np.ones((n, self.count), dtype=bool)
        self._arena_of_item = np.broadcast_to(np.arange(n)[:, None], (n, self.count))
        self.grid = np.zeros((n, h, w), dtype=np.int64)
        np.add.at(self.grid, (self._arena_of_item, self.item_y, self.item_x), 1)

    def step(self, world: World, t: int) -> None:
        if self.relocate_every and t > 0 and t % self.relocate_every == 0:
            self.centers = self._new_centers(world)
        if self.regrow_rate <= 0.0 or self.count == 0:
            return
        # Fixed number of draws per step and arena (see FoodStimulus.step).
        u = world.per_arena(self.rng.random, self.count)
        offsets = world.per_arena(lambda s: self.rng.normal(size=s), self.count, 2)
        regrow = (u < self.regrow_rate * seasonal(self, "regrow_rate")) & ~self.present
        if not regrow.any():
            return
        nx, ny = self._around_centers(world, offsets)
        self.item_x[regrow], self.item_y[regrow] = nx[regrow], ny[regrow]
        self.present[regrow] = True
        np.add.at(self.grid, (self._arena_of_item[regrow], ny[regrow], nx[regrow]), 1)
