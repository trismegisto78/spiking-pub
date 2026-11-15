import csv

import pytest

from spiking_evo.config import ConfigError, config_from_dict
from spiking_evo.genetics.population import Evolution

PHASES = [
    {"from_generation": 2, "name": "hazards",
     "set": {"stimuli": [{"type": "food", "count": 12}, {"type": "hazard", "count": 2}]}},
    {"from_generation": 4, "name": "farming",
     "set": {"network": {"action_outputs": 1},
             "stimuli": [{"type": "food", "count": 12}, {"type": "hazard", "count": 2},
                         {"type": "terrain", "fertile_zones": 2, "grow_time": 5}]}},
]  # fmt: skip


def phased(tiny_config, **kw):
    return tiny_config(phases=PHASES, genetics={"generations": 6}, **kw)


def test_phase_configs_and_validation(tiny_config):
    evo = Evolution(phased(tiny_config))
    assert [(s, n) for s, n, _ in evo.phases] == [(0, "base"), (2, "hazards"), (4, "farming")]
    assert [evo.phase_at(g) for g in range(6)] == [0, 0, 1, 1, 2, 2]
    # phases accumulate: a later phase keeps what an earlier one set
    cfg = phased(tiny_config)
    cfg.phases.append({"from_generation": 5, "name": "later", "set": {"agent": {"move_cost": 0.5}}})
    later = Evolution(cfg).phases[-1][2]
    assert later.network.action_outputs == 1 and later.agent.move_cost == 0.5
    with pytest.raises(ConfigError, match="increase"):
        config_from_dict({"phases": [{"from_generation": 3}, {"from_generation": 3}]})
    with pytest.raises(ConfigError, match="cannot change seed"):
        config_from_dict({"phases": [{"from_generation": 3, "set": {"seed": 1}}]})
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"phases": [{"from_generation": 3, "set": {"wrold": {}}}]})


def test_population_carries_over_phases(tiny_config, tmp_path):
    evo = Evolution(phased(tiny_config))
    final = evo.run(out_root=tmp_path)
    assert final[0].n_inputs == len(evo.input_names) == 17 and final[0].n_outputs == 5
    with open(tmp_path / "tiny/3/phases.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["generation"], r["phase"]) for r in rows] == [
        ("0", "base"), ("2", "hazards"), ("4", "farming")]  # fmt: skip
    with open(tmp_path / "tiny/3/stimuli.csv", newline="") as f:
        stats = {(r["generation"], r["stat"]) for r in csv.DictReader(f)}
    assert ("1", "hazard.steps_inside") not in stats and ("3", "hazard.steps_inside") in stats
    assert ("5", "terrain.planted") in stats


def test_resume_across_a_phase_change(tiny_config, tmp_path):
    full = Evolution(phased(tiny_config)).run(out_root=tmp_path / "full")
    cfg = phased(tiny_config)
    cfg.genetics.generations = 3  # stop inside phase "hazards"
    Evolution(cfg).run(out_root=tmp_path / "part")
    resumed = Evolution(phased(tiny_config)).run(out_root=tmp_path / "part", resume=True)
    assert all(a.equals(b) for a, b in zip(full, resumed, strict=True))


def test_init_from_another_experiment(tiny_config, tmp_path):
    Evolution(tiny_config()).run(out_root=tmp_path)  # food only
    cfg = tiny_config(experiment="next", stimuli=PHASES[1]["set"]["stimuli"],
                      network={"action_outputs": 1})  # fmt: skip
    evo = Evolution(cfg)
    pop = evo.run(out_root=tmp_path, init_from=tmp_path / "tiny/3")
    assert pop[0].n_outputs == 5 and pop[0].n_inputs == len(evo.input_names)
