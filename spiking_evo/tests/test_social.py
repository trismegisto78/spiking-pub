"""Stage E0: group arenas, sensing the others, patchy food."""

import numpy as np
import pytest

from spiking_evo.config import config_from_dict
from spiking_evo.config.schema import AgentConfig, WorldConfig
from spiking_evo.core.world import World
from spiking_evo.genetics.population import Evolution
from spiking_evo.stimuli.food_patch import FoodPatchStimulus
from spiking_evo.stimuli.others import OthersStimulus

CH = ["front", "left", "right", "back", "here"]


def make_world(stimuli, n, arenas="groups", group_size=3, size=9, keys=("test",)):
    cfg = WorldConfig(
        width=size, height=size, arenas=arenas, group_size=group_size, same_start=False
    )
    w = World(cfg, AgentConfig(metabolic_cost=0.0, move_cost=0.0), stimuli, n)
    w.reset(0, *keys)
    return w


def put(w, i, x, y, heading=0, arena=None):
    a = w.agents
    a.x[i], a.y[i], a.heading[i] = x, y, heading
    if arena is not None:
        a.arena[i] = arena


# -- group arenas --------------------------------------------------------------------
def test_groups_partition_agents_into_arenas_of_group_size():
    w = make_world([], 7, group_size=3)
    assert w.n_arenas == 3
    counts = np.bincount(w.agents.arena, minlength=3)
    assert sorted(counts) == [1, 3, 3]


def test_group_assignment_is_random_per_episode_and_reproducible():
    a = make_world([], 12, group_size=4, keys=("ep", 0)).agents.arena.copy()
    b = make_world([], 12, group_size=4, keys=("ep", 0)).agents.arena.copy()
    c = make_world([], 12, group_size=4, keys=("ep", 1)).agents.arena.copy()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_per_arena_draws_one_row_per_group():
    w = make_world([], 7, group_size=3)
    rng = np.random.default_rng(0)
    assert w.per_arena(rng.random, 5).shape == (3, 5)


def test_groups_cannot_be_split_across_processes(tiny_config):
    cfg = tiny_config(world={"arenas": "groups", "group_size": 4, "same_start": False})
    with pytest.raises(ValueError, match="isolated"):
        Evolution(cfg, workers=2)


def test_config_validates_groups():
    with pytest.raises(ValueError, match="arenas"):
        config_from_dict({"world": {"arenas": "teams"}})
    with pytest.raises(ValueError, match="group_size"):
        config_from_dict({"world": {"arenas": "groups", "group_size": 0}})


def test_evolution_with_groups_is_deterministic(tiny_config, tmp_path):
    cfg = tiny_config(
        world={"arenas": "groups", "group_size": 4, "same_start": False},
        stimuli=[
            {"type": "food_patch", "count": 12, "patches": 2, "regrow_rate": 0.05},
            {"type": "others"},
        ],
        genetics={"generations": 2},
    )
    a = Evolution(cfg).run(out_root=tmp_path / "a")
    b = Evolution(cfg).run(out_root=tmp_path / "b")
    assert all(x.equals(y) for x, y in zip(a, b, strict=True))


# -- others ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("heading", "other_xy", "expected"),
    [
        (0, (4, 2), "front"),
        (0, (2, 4), "left"),
        (0, (6, 4), "right"),
        (0, (4, 6), "back"),
        (1, (4, 2), "left"),
    ],
)
def test_others_sectors_are_egocentric(heading, other_xy, expected):
    others = OthersStimulus(sense_radius=3, channels=CH)
    w = make_world([others], 2, group_size=2)
    put(w, 0, 4, 4, heading, arena=0)
    put(w, 1, *other_xy, arena=0)
    obs = others.sense(w.agents, w)[0]
    assert obs[CH.index(expected)] == pytest.approx(1 - np.exp(-0.5))  # distance 2
    assert np.count_nonzero(obs) == 1


def test_others_in_other_arenas_dead_or_far_are_not_sensed():
    others = OthersStimulus(sense_radius=2, channels=CH)
    w = make_world([others], 4, group_size=2)
    put(w, 0, 4, 4, 0, arena=0)
    put(w, 1, 4, 3, arena=1)  # adjacent, other arena
    put(w, 2, 4, 1, arena=0)  # same arena, distance 3 > radius
    put(w, 3, 5, 4, arena=0)  # same arena, adjacent, but dead
    w.agents.alive[3] = False
    assert not others.sense(w.agents, w)[0].any()


def test_others_here_counts_only_other_agents():
    others = OthersStimulus(sense_radius=2, channels=CH)
    w = make_world([others], 3, group_size=3)
    for i in range(3):
        put(w, i, 1 + 3 * i, 1, arena=0)
    assert others.sense(w.agents, w)[:, CH.index("here")].tolist() == [0, 0, 0]
    put(w, 1, 1, 1, arena=0)
    assert others.sense(w.agents, w)[:, CH.index("here")].tolist() == [1, 1, 0]


def test_others_with_zero_gain_is_a_silent_control():
    others = OthersStimulus(sense_radius=3, sense_gain=0.0, channels=CH[:4])
    w = make_world([others], 3, group_size=3)
    put(w, 0, 4, 4, 0, arena=0)
    put(w, 1, 4, 3, arena=0)
    assert not others.sense(w.agents, w).any()


def test_isolated_agents_sense_no_others():
    others = OthersStimulus(sense_radius=3)
    w = make_world([others], 3, arenas="isolated")
    for i in range(3):
        put(w, i, 4, 4 + i)
    assert not others.sense(w.agents, w).any()


# -- food patches ------------------------------------------------------------------------
def torus_dist(x, y, cx, cy, size):
    dx = np.minimum(abs(x - cx), size - abs(x - cx))
    dy = np.minimum(abs(y - cy), size - abs(y - cy))
    return np.hypot(dx, dy)


def test_food_patch_items_cluster_around_centers():
    food = FoodPatchStimulus(count=60, patches=1, patch_radius=1.0)
    make_world([food], 1, arenas="isolated", size=31)
    ((cx, cy),) = food.centers[0]
    d = torus_dist(food.item_x[0], food.item_y[0], cx, cy, 31)
    assert d.max() <= 5.0
    assert food.grid.sum() == 60


def test_food_patch_regrows_near_centers_and_relocates():
    food = FoodPatchStimulus(
        count=40, patches=2, patch_radius=1.0, regrow_rate=1.0, relocate_every=10
    )
    w = make_world([food], 1, arenas="isolated", size=41)
    before = food.centers.copy()
    food.present[:] = False  # everything eaten
    food.grid[:] = 0
    for t in range(1, 11):
        food.step(w, t)
    assert not np.array_equal(food.centers, before)  # relocated at t = 10
    food.present[:] = False
    food.grid[:] = 0
    food.step(w, 11)  # regrows around the new centers
    near = np.zeros(40, dtype=bool)
    for cx, cy in food.centers[0]:
        near |= torus_dist(food.item_x[0], food.item_y[0], cx, cy, 41) <= 5.0
    assert food.present[0].all() and near.all()
    assert food.grid.sum() == 40


def test_food_patch_is_eaten_like_food():
    food = FoodPatchStimulus(count=10, patches=1, patch_radius=0.0, energy_value=5.0)
    w = make_world([food], 1, arenas="isolated", size=9)
    ((cx, cy),) = food.centers[0]
    put(w, 0, int(cx), int(cy))
    assert food.interact(w.agents, w)[0] == pytest.approx(50.0)  # all 10 on the center
    assert food.grid.sum() == 0 and not food.present.any()
