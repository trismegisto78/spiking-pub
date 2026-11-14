import csv

import pytest
import yaml

from spiking_evo.config import config_to_dict
from spiking_evo.genetics.population import Evolution
from spiking_evo.sweep import available_cpus, parse_seeds, run_sweep


def test_parse_seeds():
    assert parse_seeds("1-3,7, 9-10") == [1, 2, 3, 7, 9, 10]
    with pytest.raises(ValueError):
        parse_seeds("1-3,2")


def test_available_cpus():
    assert available_cpus() >= 1


def test_resume_after_truncated_csv_row(tiny_config, tmp_path):
    full = Evolution(tiny_config(genetics={"generations": 4})).run(out_root=tmp_path / "a")
    Evolution(tiny_config(genetics={"generations": 2})).run(out_root=tmp_path / "b")
    gens = tmp_path / "b/tiny/3/generations.csv"
    with gens.open("a") as f:
        f.write("2,12.5,3")  # machine killed while writing generation 2
    resumed = Evolution(tiny_config(genetics={"generations": 4})).run(
        out_root=tmp_path / "b", resume=True
    )
    assert all(x.equals(y) for x, y in zip(full, resumed, strict=True))
    with gens.open(newline="") as f:
        assert [r["generation"] for r in csv.DictReader(f)] == ["0", "1", "2", "3"]


def test_sweep_runs_resumes_and_skips(tiny_config, tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(config_to_dict(tiny_config())))
    out = str(tmp_path / "runs")
    rows = run_sweep([str(cfg_path)], [1, 2], out=out, jobs=2, poll_s=0.1)
    assert [r["status"] for r in rows] == ["done", "done"]
    assert all(r["last_generation"] == "3" for r in rows)
    assert "gen    3" in (tmp_path / "runs/tiny/1/run.log").read_text()
    # A sweep launched again skips finished runs and extends a longer one.
    rows = run_sweep([str(cfg_path)], [1, 2], out=out, jobs=2, generations=6, poll_s=0.1)
    assert [r["last_generation"] for r in rows] == ["5", "5"]
    rows = run_sweep([str(cfg_path)], [1, 2], out=out, jobs=2, generations=6, poll_s=0.1)
    assert [r["status"] for r in rows] == ["skipped (finished)"] * 2
