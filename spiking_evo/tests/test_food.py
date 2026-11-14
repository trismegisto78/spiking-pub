import numpy as np
import pytest

from spiking_evo.config.schema import AgentConfig, WorldConfig
from spiking_evo.core.world import World
from spiking_evo.stimuli.food import FoodStimulus
from spiking_evo.stimuli.registry import available, build_stimuli, build_stimulus


def world_with_food(food, n=1, arenas="isolated", wrap=True, size=9):
    cfg = WorldConfig(width=size, height=size, wrap=wrap, arenas=arenas)
    w = World(cfg, AgentConfig(metabolic_cost=0.0, move_cost=0.0), [food], n)
    w.reset(0, "test")
    return w


def place_food(food, world, cells):
    food.grid[:] = 0
    food.present[:] = False
    used = {}
    for a, x, y in cells:
        i = used[a] = used.get(a, -1) + 1
        food.item_x[a, i], food.item_y[a, i] = x, y
        food.present[a, i] = True
        food.grid[a, y, x] += 1


def put_agent(world, i, x, y, heading):
    world.agents.x[i], world.agents.y[i], world.agents.heading[i] = x, y, heading


CH = ["front", "left", "right", "back", "here"]


@pytest.mark.parametrize(
    ("heading", "food_xy", "expected"),
    [
        (0, (4, 2), "front"),  # facing north, food north
        (0, (2, 4), "left"),  # facing north, food west
        (0, (6, 4), "right"),
        (0, (4, 6), "back"),
        (1, (6, 4), "front"),  # facing east, food east
        (1, (4, 2), "left"),  # facing east, food north
        (2, (2, 4), "right"),  # facing south, food west is on the right
        (3, (4, 6), "left"),  # facing west, food south is on the left
    ],
)
def test_sense_sectors_are_egocentric(heading, food_xy, expected):
    food = FoodStimulus(count=1, sense_radius=3, channels=CH)
    w = world_with_food(food)
    place_food(food, w, [(0, *food_xy)])
    put_agent(w, 0, 4, 4, heading)
    obs = food.sense(w.agents, w)[0]
    assert obs[CH.index(expected)] == pytest.approx(1 - np.exp(-0.5))  # distance 2
    assert np.count_nonzero(obs) == 1


def test_sense_here_and_radius():
    food = FoodStimulus(count=2, sense_radius=2, channels=CH)
    w = world_with_food(food)
    place_food(food, w, [(0, 4, 4), (0, 4, 1)])  # on the agent, and 3 cells north
    put_agent(w, 0, 4, 4, 0)
    obs = food.sense(w.agents, w)[0]
    assert obs.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_sense_through_wrap_and_walls():
    food = FoodStimulus(count=1, sense_radius=2, channels=["front"])
    w = world_with_food(food, wrap=True)
    place_food(food, w, [(0, 0, 8)])
    put_agent(w, 0, 0, 0, 0)  # facing north from the top row: food is 1 cell away via wrap
    assert food.sense(w.agents, w)[0, 0] == pytest.approx(1 - np.exp(-1))
    food2 = FoodStimulus(count=1, sense_radius=2, channels=["front"])
    w2 = world_with_food(food2, wrap=False)
    place_food(food2, w2, [(0, 0, 8)])
    put_agent(w2, 0, 0, 0, 0)
    assert food2.sense(w2.agents, w2)[0, 0] == 0.0


def test_eating_gives_energy_and_removes_food():
    food = FoodStimulus(count=1, energy_value=5.0)
    w = world_with_food(food, n=2)
    place_food(food, w, [(0, 3, 3), (1, 3, 3)])
    put_agent(w, 0, 3, 3, 0)
    put_agent(w, 1, 3, 3, 0)
    w.agents.alive[1] = False
    delta = food.interact(w.agents, w)
    assert delta.tolist() == [5.0, 0.0]  # dead agents do not eat
    assert food.remaining().tolist() == [0, 1]
    assert food.interact(w.agents, w).tolist() == [0.0, 0.0]


def test_shared_arena_food_is_eaten_once():
    food = FoodStimulus(count=1, energy_value=5.0)
    w = world_with_food(food, n=3, arenas="shared")
    assert w.n_arenas == 1
    place_food(food, w, [(0, 2, 2)])
    for i in range(3):
        put_agent(w, i, 2, 2, 0)
    delta = food.interact(w.agents, w)
    assert sorted(delta.tolist()) == [0.0, 0.0, 5.0]


def test_regrowth():
    food = FoodStimulus(count=10, regrow_rate=1.0)
    w = world_with_food(food)
    food.present[:] = False
    food.grid[:] = 0
    food.step(w, 0)
    assert food.remaining().tolist() == [10]
    assert food.grid.sum() == 10
    fixed = FoodStimulus(count=10, regrow_rate=0.0)
    w2 = world_with_food(fixed)
    fixed.present[:] = False
    fixed.step(w2, 0)
    assert fixed.remaining().tolist() == [0]


def test_layouts():
    food = FoodStimulus(count=20, shared_layout=True)
    world_with_food(food, n=4)
    assert (food.grid == food.grid[0]).all() and food.grid.sum() == 80
    food2 = FoodStimulus(count=20, shared_layout=False)
    world_with_food(food2, n=4)
    assert not (food2.grid == food2.grid[0]).all()
    assert (food2.grid.sum(axis=(1, 2)) == 20).all() and food2.grid.max() == 1


def test_registry():
    assert "food" in available()
    stim = build_stimulus({"type": "food", "count": 3, "channels": ["front"]})
    assert isinstance(stim, FoodStimulus) and stim.count == 3 and stim.channels == ["front"]
    with pytest.raises(ValueError, match="unknown stimulus"):
        build_stimulus({"type": "lava"})
    with pytest.raises(ValueError, match="unknown parameter"):
        build_stimulus({"type": "food", "colour": "red"})
    with pytest.raises(ValueError, match="unique"):
        build_stimuli([{"type": "food"}, {"type": "food"}])
    assert len(build_stimuli([{"type": "food"}, {"type": "food", "name": "food2"}])) == 2
