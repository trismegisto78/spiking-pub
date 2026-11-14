import csv
import json

import numpy as np

from spiking_evo.cli import main
from spiking_evo.fitness.evaluator import BehaviourStats, compute_terms, total_fitness
from spiking_evo.genetics.genome import Genome
from spiking_evo.genetics.population import Evolution, n_input_channels
from spiking_evo.seeding import derive_rng


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_derive_rng_is_stable_and_keyed():
    a = derive_rng(1, "eval", 3).random(4)
    assert np.array_equal(a, derive_rng(1, "eval", 3).random(4))
    assert not np.array_equal(a, derive_rng(1, "eval", 4).random(4))
    assert not np.array_equal(a, derive_rng(2, "eval", 3).random(4))


def test_input_size_follows_declared_channels(tiny_config):
    cfg = tiny_config(stimuli=[{"type": "food", "channels": ["front", "here"]}])
    n, names = n_input_channels(cfg)
    assert n == 3 and names == ["food.front", "food.here", "bias"]


def test_fitness_terms():
    from spiking_evo.config.schema import FitnessConfig

    cfg = FitnessConfig()
    cfg.exploration.weight = 0.1
    s = BehaviourStats(
        energy_gained=np.array([10.0, 0.0, 5.0]),
        final_energy=np.zeros(3),
        firing_rate_hz=np.array([10.0, 0.0, 60.0]),
        noninput_spikes=np.array([5, 0, 100]),
        visited_cells=np.array([10.0, 1.0, 20.0]),
        steps_alive=np.zeros(3),
    )
    t = compute_terms(cfg, s)
    assert t["energy"].tolist() == [10.0, 0.0, 5.0]
    assert t["firing_penalty"].tolist() == [0.0, 0.0, -1.0]  # 0.5 * (60 - 20) / 20
    assert t["silence_penalty"].tolist() == [0.0, -0.5, 0.0]
    assert np.allclose(t["exploration"], [1.0, 0.1, 2.0])
    assert np.allclose(total_fitness(t), [11.0, -0.4, 6.0])


def test_same_seed_same_final_genome(tiny_config, tmp_path):
    cfg = tiny_config()
    pop_a = Evolution(cfg).run(out_root=tmp_path / "a")
    pop_b = Evolution(cfg).run(out_root=tmp_path / "b")
    assert all(a.equals(b) for a, b in zip(pop_a, pop_b, strict=True))
    rows_a = read_csv(tmp_path / "a/tiny/3/generations.csv")
    rows_b = read_csv(tmp_path / "b/tiny/3/generations.csv")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows]  # noqa: E731
    assert strip(rows_a) == strip(rows_b)
    pop_c = Evolution(tiny_config(seed=4)).run(out_root=tmp_path / "c")
    assert not all(a.equals(c) for a, c in zip(pop_a, pop_c, strict=True))


def test_generation_is_reproducible_in_isolation(tiny_config):
    evo = Evolution(tiny_config())
    pop = evo.initial_population()
    f1 = evo.evaluate(pop, 7).fitness
    evo.evaluate(pop, 2)
    assert np.array_equal(f1, evo.evaluate(pop, 7).fitness)


def test_resume_matches_uninterrupted_run(tiny_config, tmp_path):
    full = Evolution(tiny_config(genetics={"generations": 5})).run(out_root=tmp_path / "full")
    Evolution(tiny_config(genetics={"generations": 2})).run(out_root=tmp_path / "part")
    resumed = Evolution(tiny_config(genetics={"generations": 5})).run(
        out_root=tmp_path / "part", resume=True
    )
    assert all(a.equals(b) for a, b in zip(full, resumed, strict=True))
    gens = [int(r["generation"]) for r in read_csv(tmp_path / "part/tiny/3/generations.csv")]
    assert gens == [0, 1, 2, 3, 4]


def test_outputs_on_disk(tiny_config, tmp_path):
    for encoder in ("rate", "latency", "current"):
        cfg = tiny_config(experiment=f"tiny_{encoder}", network={"encoder": encoder})
        Evolution(cfg).run(out_root=tmp_path)
        d = tmp_path / f"tiny_{encoder}" / "3"
        rows = read_csv(d / "generations.csv")
        assert len(rows) == 4
        assert {
            "fitness_mean",
            "fitness_max",
            "fitness_min",
            "fitness_var",
            "n_species",
            "firing_rate_mean",
            "energy_mean",
        } <= set(rows[0])
        terms = read_csv(d / "terms.csv")
        assert len(terms) == 4 and "firing_penalty_mean" in terms[0]
        snaps = sorted(p.name for p in (d / "snapshots").iterdir())
        assert snaps == ["gen_0000.json", "gen_0002.json", "gen_0003.json"]
        snap = json.loads((d / "snapshots/gen_0003.json").read_text())
        Genome.from_dict(snap["genome"])
        cfg_text = (d / "config.yaml").read_text()
        assert "commit:" in cfg_text and f"experiment: tiny_{encoder}" in cfg_text


def test_cli_run_and_evaluate(tiny_config, tmp_path, capsys):
    import yaml

    from spiking_evo.config import config_to_dict

    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(config_to_dict(tiny_config())))
    assert main(["run", str(cfg_path), "--out", str(tmp_path), "--generations", "2"]) == 0
    snap = tmp_path / "tiny/3/snapshots/gen_0001.json"
    assert snap.exists()
    assert main(["evaluate", str(snap), "--config", str(cfg_path), "--episodes", "2"]) == 0
    assert "fitness_mean" in capsys.readouterr().out
