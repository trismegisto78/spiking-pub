import json

import numpy as np

from spiking_evo.analysis import analyze, load_runs
from spiking_evo.behaviour import behaviour, format_report
from spiking_evo.genetics.genome import Genome
from spiking_evo.genetics.population import Evolution

SEASONS = [{"name": "warm", "length": 20}, {"name": "cold", "length": 20,
                                             "living_cost": 0.1}]  # fmt: skip


def seasonal_config(tiny_config, **kw):
    return tiny_config(
        network={"action_outputs": 1},
        stimuli=[
            {"type": "season", "seasons": SEASONS},
            {"type": "food", "count": 12, "regrow_rate": 0.05},
            {"type": "terrain", "fertile_zones": 3, "grow_time": 5},
        ],
        **kw,
    )


def test_analyze_across_seeds(tiny_config, tmp_path):
    for seed in (1, 2):
        Evolution(seasonal_config(tiny_config, seed=seed)).run(out_root=tmp_path / "runs")
    exps = load_runs(tmp_path / "runs")
    exp = exps["tiny"]
    assert exp.seeds == [1, 2] and exp.gen["fitness_mean"].shape == (2, 4)
    assert exp.season_names() == ["warm", "cold"]
    res = analyze(tmp_path / "runs", tmp_path / "report", tail=2)
    text = res["summary"].read_text()
    assert "## tiny" in text and "Seed: 2 (1, 2)" in text and "| cold |" in text
    assert {p.name for p in res["plots"]} >= {"fitness.png", "tiny_seasons.png"}


def test_behaviour_conditions(tiny_config):
    cfg = seasonal_config(tiny_config)
    evo = Evolution(cfg)
    g = evo.initial_population()[0]
    g = Genome.from_dict(json.loads(json.dumps(g.to_dict())))
    rep = behaviour(cfg, g, episodes=3)
    assert list(rep) == ["normal", "blind:season", "blind:food", "blind:terrain"]
    assert {"fitness", "alive_at_end", "terrain.planted", "season.energy_cold"} <= set(
        rep["normal"]
    )
    # deterministic: the same call gives the same numbers
    assert behaviour(cfg, g, episodes=3, conditions=["normal"])["normal"] == rep["normal"]
    assert "| fitness |" in format_report(rep)
    assert all(np.isfinite(v[0]) for v in rep["normal"].values())
