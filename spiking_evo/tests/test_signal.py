"""Stage E1: the `signal` stimulus and groups of clones."""

import numpy as np
import pytest

from spiking_evo.config import config_from_dict
from spiking_evo.config.schema import AgentConfig, WorldConfig
from spiking_evo.core.world import World
from spiking_evo.genetics.population import Evolution
from spiking_evo.stimuli.signal import SignalStimulus

CH = ["front", "left", "right", "back"]


def make_world(stimuli, n, group_size=4, shuffle=True, arenas="groups"):
    cfg = WorldConfig(
        width=9, height=9, arenas=arenas, group_size=group_size, group_shuffle=shuffle,
        same_start=False,
    )  # fmt: skip
    w = World(cfg, AgentConfig(metabolic_cost=0.0, move_cost=0.0), stimuli, n, n_actions=1)
    w.reset(0, "test")
    return w


def put(w, i, x, y, heading=0, arena=0):
    a = w.agents
    a.x[i], a.y[i], a.heading[i], a.arena[i] = x, y, heading, arena


def emit(w, sig, who):
    w.agents.action[:] = False
    w.agents.action[who, 0] = True
    return sig.interact(w.agents, w)


# -- signal ---------------------------------------------------------------------------
def test_signal_is_heard_next_step_in_egocentric_sectors():
    sig = SignalStimulus(action=0, sense_radius=3, emit_cost=0.1)
    w = make_world([sig], 3)
    put(w, 0, 4, 4, heading=0)
    put(w, 1, 4, 2)  # 2 cells north of agent 0
    put(w, 2, 2, 4)  # 2 cells west of agent 0
    assert not sig.sense(w.agents, w).any()  # nobody has emitted yet
    delta = emit(w, sig, [1])
    assert delta.tolist() == pytest.approx([0.0, -0.1, 0.0])  # emitting costs energy
    obs = sig.sense(w.agents, w)
    assert obs[0, CH.index("front")] == pytest.approx(1 - np.exp(-0.5))
    assert np.count_nonzero(obs[0]) == 1
    # agent 2 faces north, agent 1 is 2 east and 2 north: diagonals belong to "front"
    assert obs[2, CH.index("front")] > 0 and np.count_nonzero(obs[2]) == 1
    assert not obs[1].any()  # an emitter does not hear itself


def test_signal_only_within_arena_radius_and_from_the_living():
    sig = SignalStimulus(action=0, sense_radius=2)
    w = make_world([sig], 4, group_size=2)
    put(w, 0, 4, 4)
    put(w, 1, 4, 3, arena=1)  # adjacent, other arena
    put(w, 2, 4, 1)  # same arena, distance 3 > radius
    put(w, 3, 5, 4)  # same arena, adjacent, dead
    w.agents.alive[3] = False
    emit(w, sig, [1, 2, 3])
    assert not sig.sense(w.agents, w)[0].any()


def test_signal_lasts_one_step_and_mute_control_hears_nothing():
    sig = SignalStimulus(action=0, sense_radius=3)
    mute = SignalStimulus(action=0, sense_radius=3, sense_gain=0.0, name="mute")
    w = make_world([sig, mute], 2)
    put(w, 0, 4, 4)
    put(w, 1, 4, 3)
    emit(w, sig, [1])
    emit(w, mute, [1])
    assert sig.sense(w.agents, w)[0].any()
    assert not mute.sense(w.agents, w).any()
    emit(w, sig, [])
    assert not sig.sense(w.agents, w).any()


def test_signal_counts_emissions_while_fed_and_unfed():
    sig = SignalStimulus(action=0, fed_window=2)
    w = make_world([sig], 2)
    a = w.agents
    # step 1: agent 0 eats and emits; agent 1 does neither
    a.energy_gained[0] += 5.0
    emit(w, sig, [0])
    # step 2: nobody eats; agent 0 is still "fed" (window 2), emits; agent 1 emits unfed
    emit(w, sig, [0, 1])
    # step 3: agent 0 no longer fed, emits
    emit(w, sig, [0])
    s = sig.episode_stats()
    assert s["emitted"].tolist() == [3, 1]
    assert s["steps_fed"].tolist() == [2, 0]
    assert s["emitted_fed"].tolist() == [2, 0]
    assert s["steps_unfed"].tolist() == [1, 3]
    assert s["emitted_unfed"].tolist() == [1, 1]


# -- groups of clones ---------------------------------------------------------------------
def test_unshuffled_groups_keep_consecutive_agents_together():
    w = make_world([], 7, group_size=3, shuffle=False)
    assert w.agents.arena.tolist() == [0, 0, 0, 1, 1, 1, 2]


def clone_cfg(tiny_config, **genetics):
    return tiny_config(
        world={"arenas": "groups", "group_size": 3, "group_shuffle": False, "same_start": False},
        stimuli=[
            {"type": "food_patch", "count": 12, "patches": 2, "regrow_rate": 0.05},
            {"type": "signal", "action": 0},
        ],
        network={"action_outputs": 1},
        genetics={"population": 4, "elitism": 1, "clones": 3, **genetics},
    )


def test_clones_evaluate_each_genome_as_a_group(tiny_config):
    evo = Evolution(clone_cfg(tiny_config))
    pop = evo.initial_population()
    assert len(pop) == 4
    ev = evo.evaluate(pop, 0)
    assert ev.fitness.shape == (4,)
    assert ev.stimulus_stats["signal.emitted"].shape == (4,)
    # one genome = mean of its 3 clones evaluated in one arena
    full = evo.evaluate_chunk([g for g in pop for _ in range(3)], 0, 0, 12)
    assert np.allclose(ev.fitness, full.fitness.reshape(4, 3).mean(axis=1))


def test_clones_run_is_deterministic(tiny_config, tmp_path):
    cfg = clone_cfg(tiny_config, generations=2)
    a = Evolution(cfg).run(out_root=tmp_path / "a")
    b = Evolution(cfg).run(out_root=tmp_path / "b")
    assert all(x.equals(y) for x, y in zip(a, b, strict=True))


def test_config_validates_clones():
    ok = {"world": {"arenas": "groups", "group_size": 5, "group_shuffle": False}}
    config_from_dict({**ok, "genetics": {"clones": 5}})
    with pytest.raises(ValueError, match="clones"):
        config_from_dict({**ok, "genetics": {"clones": 4}})  # group_size must equal clones
    with pytest.raises(ValueError, match="clones"):
        config_from_dict({"genetics": {"clones": 5}})  # needs group arenas
    with pytest.raises(ValueError, match="clones"):
        bad = {"world": {"arenas": "groups", "group_size": 5}}  # shuffled groups
        config_from_dict({**bad, "genetics": {"clones": 5}})
