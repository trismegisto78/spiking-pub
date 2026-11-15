import numpy as np
import pytest

from spiking_evo.config.schema import AgentConfig, WorldConfig
from spiking_evo.core.world import World
from spiking_evo.stimuli.hazard import HazardStimulus
from spiking_evo.stimuli.registry import available, build_stimulus

CH = ["front", "left", "right", "back", "here"]


def make_world(stimuli, n=1, wrap=True, size=11):
    cfg = WorldConfig(width=size, height=size, wrap=wrap)
    w = World(cfg, AgentConfig(metabolic_cost=0.0, move_cost=0.0), stimuli, n)
    w.reset(0, "test")
    return w


def single_cell(hz, w, x, y):
    hz.grid[:] = 0.0
    hz.grid[:, y, x] = 1.0


def put(w, i, x, y, h):
    w.agents.x[i], w.agents.y[i], w.agents.heading[i] = x, y, h


def test_registered():
    assert "hazard" in available()
    hz = build_stimulus({"type": "hazard", "count": 2, "damage": 3.0})
    assert isinstance(hz, HazardStimulus) and hz.damage == 3.0


def test_zone_geometry_and_wrap():
    hz = HazardStimulus(count=1, radius=1.0)
    make_world([hz])
    cx, cy = hz.centers[0, 0]
    assert hz.grid[0].sum() == 5  # centre + 4 neighbours (radius 1, euclidean)
    assert hz.grid[0, cy, cx] == 1.0
    assert hz.grid[0, (cy + 1) % 11, cx] == 1.0 and hz.grid[0, (cy + 1) % 11, (cx + 1) % 11] == 0.0


def test_shared_and_private_layouts():
    a = HazardStimulus(count=3, shared_layout=True)
    make_world([a], n=4)
    assert (a.grid == a.grid[0]).all()
    b = HazardStimulus(count=3, shared_layout=False)
    make_world([b], n=4)
    assert not (b.grid == b.grid[0]).all()


def test_damage_only_inside_and_alive():
    hz = HazardStimulus(count=1, damage=2.5)
    w = make_world([hz], n=3)
    single_cell(hz, w, 5, 5)
    put(w, 0, 5, 5, 0)
    put(w, 1, 6, 5, 0)
    put(w, 2, 5, 5, 0)
    w.agents.alive[2] = False
    assert hz.interact(w.agents, w).tolist() == [-2.5, 0.0, 0.0]


@pytest.mark.parametrize(
    ("heading", "xy", "channel"),
    [(0, (5, 3), "front"), (0, (3, 5), "left"), (1, (5, 3), "left"), (2, (5, 7), "front")],
)
def test_sensing_is_egocentric(heading, xy, channel):
    hz = HazardStimulus(count=1, channels=CH, sense_gain=1.0)
    w = make_world([hz])
    single_cell(hz, w, *xy)
    put(w, 0, 5, 5, heading)
    obs = hz.sense(w.agents, w)[0]
    assert obs[CH.index(channel)] == pytest.approx(1 - np.exp(-0.5))
    assert np.count_nonzero(obs) == 1


def test_food_and_hazard_channels_are_separate():
    w = make_world([build_stimulus({"type": "food", "count": 3}), HazardStimulus(count=1)], size=11)
    assert w.channel_names == [
        "food.front",
        "food.left",
        "food.right",
        "food.here",
        "hazard.front",
        "hazard.left",
        "hazard.right",
        "hazard.here",
    ]


def test_standing_in_hazard_drains_energy_over_episode():
    class Still:
        def reset(self):
            pass

        def act(self, obs, active):
            n = obs.shape[0]
            return np.zeros(n, int), np.zeros(n, int)

    hz = HazardStimulus(count=1, radius=20.0, damage=1.0)  # the whole grid is hazard
    w = make_world([hz])
    res = w.run_episode(Still(), 0, "ep")
    assert res.energy_gained[0] == -50.0  # initial energy 50, dies after 50 steps
    assert not res.alive[0]
