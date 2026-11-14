import ast
from pathlib import Path

import numpy as np
import pytest

from spiking_evo.config.schema import AgentConfig, WorldConfig
from spiking_evo.core.clock import Clock
from spiking_evo.core.world import World


class Scripted:
    """Controller replaying fixed actions."""

    def __init__(self, forward, turn):
        self.forward, self.turn = forward, turn

    def reset(self):
        pass

    def act(self, obs, active):
        n = obs.shape[0]
        return np.full(n, self.forward), np.full(n, self.turn)


class ConstantDrain:
    """Minimal stimulus used to check that the core is stimulus-agnostic."""

    name = "drain"
    channels = ["level"]

    def __init__(self, amount):
        self.amount = amount
        self.steps = 0

    def reset(self, world, rng):
        self.steps = 0

    def step(self, world, t):
        self.steps += 1

    def sense(self, agents, world):
        return np.full((len(agents), 1), 0.5)

    def interact(self, agents, world):
        return np.where(agents.alive, -self.amount, 0.0)


def make_world(stimuli=(), n=2, **kw):
    agent = AgentConfig(
        initial_energy=10.0,
        metabolic_cost=kw.pop("metabolic_cost", 0.0),
        move_cost=kw.pop("move_cost", 0.0),
    )
    return World(WorldConfig(width=5, height=4, steps_per_episode=50, **kw), agent, stimuli, n)


def test_clock():
    c = Clock(dt_ms=0.5, steps_per_tick=4)
    c.advance()
    c.advance()
    assert (c.tick, c.snn_step, c.time_ms, c.tick_ms) == (2, 8, 4.0, 2.0)


def test_forward_movement_wraps_around():
    w = make_world()
    w.reset(0, "t")
    w.agents.x[:], w.agents.y[:], w.agents.heading[:] = 4, 0, 1  # east edge, facing east
    w.step(Scripted(1, 0))
    assert w.agents.x.tolist() == [0, 0]
    w.agents.heading[:] = 0  # north, from y=0 wraps to the bottom row
    w.step(Scripted(1, 0))
    assert w.agents.y.tolist() == [3, 3]


def test_walls_block_movement():
    w = make_world(wrap=False, move_cost=1.0)
    w.reset(0, "t")
    w.agents.x[:], w.agents.y[:], w.agents.heading[:] = 0, 0, 3  # west edge, facing west
    w.step(Scripted(1, 0))
    assert w.agents.x.tolist() == [0, 0]
    assert w.agents.energy.tolist() == [10.0, 10.0]  # no move, no move cost


def test_turn_then_move_and_costs():
    w = make_world(move_cost=0.5, metabolic_cost=0.25)
    w.reset(0, "t")
    w.agents.x[:], w.agents.y[:], w.agents.heading[:] = 2, 2, 0
    w.step(Scripted(1, 1))  # turn right (east), then move east
    assert w.agents.heading.tolist() == [1, 1]
    assert w.agents.x.tolist() == [3, 3]
    assert w.agents.energy.tolist() == [9.25, 9.25]
    w.step(Scripted(-1, -1))  # turn left (north), move backward (south)
    assert w.agents.heading.tolist() == [0, 0]
    assert w.agents.y.tolist() == [3, 3]


def test_stimulus_interface_and_death():
    drain = ConstantDrain(3.0)
    w = make_world([drain])
    res = w.run_episode(Scripted(0, 0), 0, "ep")
    assert drain.steps == 4  # episode stops early once everybody is dead
    assert res.alive.tolist() == [False, False]
    assert res.steps_alive.tolist() == [4, 4]
    assert res.energy_gained.tolist() == [-12.0, -12.0]
    assert res.final_energy.tolist() == [0.0, 0.0]
    assert w.n_channels == 1 and w.channel_names == ["drain.level"]
    assert w.sense().tolist() == [[0.0], [0.0]]  # dead agents perceive nothing


def test_reset_is_deterministic():
    w = make_world(n=3, same_start=False)
    w.reset(5, "a")
    first = (w.agents.x.copy(), w.agents.y.copy(), w.agents.heading.copy())
    w.reset(5, "a")
    assert all(
        np.array_equal(a, b)
        for a, b in zip(first, (w.agents.x, w.agents.y, w.agents.heading), strict=True)
    )


def test_visited_cells():
    w = make_world(n=1)
    w.reset(0, "t")
    w.agents.x[:], w.agents.y[:], w.agents.heading[:] = 0, 0, 1
    w.visited.fill(False)
    w._mark_visited()
    for _ in range(3):
        w.step(Scripted(1, 0))
    assert w.visited.sum() == 4


@pytest.mark.parametrize("module", ["world.py", "agent.py", "clock.py"])
def test_core_does_not_import_concrete_stimuli(module):
    src = (Path(__file__).parents[1] / "core" / module).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module and "stimuli" in node.module:
            assert node.module.endswith("stimuli.base"), node.module
