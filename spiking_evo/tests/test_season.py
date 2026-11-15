import numpy as np
import pytest

from spiking_evo.config.schema import AgentConfig, WorldConfig
from spiking_evo.core.world import World
from spiking_evo.genetics.population import Evolution, merge_evaluations
from spiking_evo.stimuli.registry import build_stimuli

SEASONS = [
    {"name": "summer", "length": 3, "modifiers": {"hazard.damage": 2.0}},
    {"name": "winter", "length": 2, "modifiers": {"food.regrow_rate": 0.0,
                                                  "terrain.growth_rate": 0.5}},
]  # fmt: skip


def make_world(specs, n=1, n_actions=1):
    cfg = WorldConfig(width=10, height=10, steps_per_episode=50)
    agent = AgentConfig(initial_energy=100.0, metabolic_cost=0.0, move_cost=0.0)
    w = World(cfg, agent, build_stimuli(specs), n, n_actions=n_actions)
    w.reset(0, "t")
    return w


class Still:
    def __init__(self, act_at=()):
        self.act_at, self.t = set(act_at), 0

    def reset(self):
        self.t = 0

    def act(self, obs, active):
        n = obs.shape[0]
        do = self.t in self.act_at
        self.t += 1
        return np.zeros(n, int), np.zeros(n, int), np.full((n, 1), do)


def test_cycle_and_sensing():
    w = make_world([{"type": "season", "seasons": SEASONS,
                     "channels": ["phase_sin", "phase_cos", "summer", "winter"]},
                    {"type": "food"}, {"type": "hazard"}, {"type": "terrain"}])  # fmt: skip
    season = w.stimuli[0]
    assert [season.season_at(t) for t in range(7)] == [0, 0, 0, 1, 1, 0, 0]
    obs = season.sense(w.agents, w)[0]
    assert obs[2:].tolist() == [1.0, 0.0] and obs[0] == pytest.approx(0.5)
    assert obs[1] == pytest.approx(1.0)
    season.step(w, 3)
    assert season.sense(w.agents, w)[0, 2:].tolist() == [0.0, 1.0]


def test_modifiers_are_applied_and_reset_between_seasons():
    w = make_world([{"type": "season", "seasons": SEASONS},
                    {"type": "food", "regrow_rate": 0.3}, {"type": "hazard", "damage": 2.0},
                    {"type": "terrain"}])  # fmt: skip
    season, food, hazard, terrain = w.stimuli
    from spiking_evo.stimuli.base import seasonal

    assert seasonal(hazard, "damage") == 2.0 and seasonal(food, "regrow_rate") == 1.0
    season.step(w, 3)  # winter
    assert seasonal(hazard, "damage") == 1.0 and seasonal(food, "regrow_rate") == 0.0
    assert seasonal(terrain, "growth_rate") == 0.5
    # hazard damage is doubled in summer
    season.step(w, 0)
    hazard.grid = np.full(hazard.grid.shape, 2.0)
    assert hazard.interact(w.agents, w)[0] == -4.0


def test_winter_slows_crop_growth():
    seasons = [{"name": "winter", "length": 1000, "modifiers": {"terrain.growth_rate": 0.5}}]
    w = make_world([{"type": "season", "seasons": seasons},
                    {"type": "terrain", "grow_time": 4, "harvest_energy": 10.0,
                     "action_cost": 0.0}])  # fmt: skip
    terrain = w.stimuli[1]
    terrain.variety = np.ones(terrain.sown.shape, dtype=np.int64)
    terrain.fertile = terrain.variety > 0
    w.agents.x[:], w.agents.y[:] = 5, 5
    ctrl = Still(act_at={0})
    gains = []
    for _ in range(10):
        before = w.agents.energy[0]
        w.step(ctrl)
        gains.append(w.agents.energy[0] - before)
    assert gains.index(10.0) == 8  # 4 steps of growth at half speed = 8 steps


def test_random_phase_and_validation():
    w = make_world([{"type": "season", "seasons": SEASONS, "random_phase": True},
                    {"type": "food"}, {"type": "hazard"}, {"type": "terrain"}])  # fmt: skip
    assert 0 <= w.stimuli[0].offset < 5
    with pytest.raises(ValueError, match="no stimulus named"):
        make_world(
            [{"type": "season", "seasons": [{"length": 5, "modifiers": {"rain.amount": 2.0}}]}]
        )
    with pytest.raises(ValueError, match="accepts"):
        make_world(
            [
                {"type": "season", "seasons": [{"length": 5, "modifiers": {"food.colour": 2.0}}]},
                {"type": "food"},
            ]
        )


def test_evolution_with_seasons_is_chunk_invariant(tiny_config):
    cfg = tiny_config(
        network={"action_outputs": 1},
        world={"same_start": False},
        stimuli=[
            {"type": "season", "seasons": SEASONS, "random_phase": True},
            {"type": "food", "count": 12, "regrow_rate": 0.1, "shared_layout": False},
            {"type": "hazard", "count": 2, "shared_layout": False},
            {"type": "terrain", "fertile_zones": 3, "grow_time": 5, "shared_layout": False},
        ],
    )
    evo = Evolution(cfg)
    assert evo.input_names[:2] == ["season.phase_sin", "season.phase_cos"]
    pop = evo.initial_population()
    full = evo.evaluate(pop, 1)
    parts = merge_evaluations(
        [evo.evaluate_chunk(pop[:7], 1, 0, 12), evo.evaluate_chunk(pop[7:], 1, 7, 12)]
    )
    assert np.array_equal(full.fitness, parts.fitness)


def test_energy_is_booked_per_season():
    seasons = [{"name": "a", "length": 2}, {"name": "b", "length": 2}]
    w = make_world([{"type": "season", "seasons": seasons},
                    {"type": "hazard", "count": 1, "radius": 20.0, "damage": 1.0}])  # fmt: skip
    ctrl = Still()
    for _ in range(5):
        w.step(ctrl)
    st = w.stimuli[0].episode_stats()
    # steps 0-1 in a, 2-3 in b, step 4 in a (booked at the next step: not yet)
    assert st["energy_a"][0] == -2.0 and st["energy_b"][0] == -2.0
    assert st["steps_a"][0] == 2 and st["steps_b"][0] == 2
