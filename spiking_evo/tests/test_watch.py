"""The `watch` GUI backend: finding runs, reading CSVs, live sessions, HTTP server."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from spiking_evo.config import config_from_dict
from spiking_evo.genetics.population import Evolution
from spiking_evo.watch.data import find_runs, read_csv_columns, run_series
from spiking_evo.watch.server import App, make_handler
from spiking_evo.watch.sim import EcosystemSession, open_session


@pytest.fixture
def runs_root(tiny_config, tmp_path):
    root = tmp_path / "runs"
    cfg = tiny_config(
        world={"arenas": "groups", "group_size": 4, "same_start": False},
        stimuli=[
            {"type": "food_patch", "count": 12, "patches": 2, "regrow_rate": 0.05},
            {"type": "others"},
        ],
        genetics={"generations": 3},
    )
    Evolution(cfg).run(out_root=root)
    return root


def test_find_runs_and_series(runs_root):
    (runs_root / "notes").mkdir()  # not a run: ignored
    runs = find_runs([runs_root])
    assert len(runs) == 1
    r = runs[0]
    assert (r["experiment"], r["seed"], r["kind"], r["source"]) == (
        "tiny",
        "3",
        "generations",
        "runs",
    )
    assert r["checkpoint"] and r["snapshots"]
    s = run_series(runs_root / "tiny" / "3")
    assert s["generations.csv"]["generation"] == [0, 1, 2]
    assert all(isinstance(v, float) for v in s["generations.csv"]["fitness_mean"])


def test_csv_reader_drops_truncated_rows_and_keeps_text(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b,c\n1,x,\n2,3")  # last row cut short
    assert read_csv_columns(p) == {"a": [1.0], "b": ["x"], "c": [None]}


@pytest.mark.parametrize("genomes", ["checkpoint", "random", "snapshot"])
def test_generational_session_frames_and_agent(runs_root, genomes):
    run = runs_root / "tiny" / "3"
    if genomes == "snapshot":
        genomes = sorted(p.name for p in (run / "snapshots").glob("*.json"))[-1]
    s = open_session({"run": str(run), "genomes": genomes})
    info = s.info()
    assert info["kind"] == "generations" and info["width"] == 12
    s.step(70)  # crosses the end of a 60-step episode
    f = s.frame(0)
    assert f["episode"] == 1 and f["tick"] == 10
    assert f["layers"]["food"]["kind"] == "food" and len(f["layers"]["food"]["data"]) == 144
    assert 1 <= len(f["agents"]) <= 4
    a = s.agent(f["agents"][0]["i"])
    assert set(a["inputs"]) == set(info["inputs"])
    b = a["brain"]
    assert len(b["spikes"]) == b["n_neurons"] and len(b["src"]) == len(b["weight"])


def test_ecosystem_session():
    cfg = config_from_dict(
        {
            "experiment": "eco_test",
            "seed": 5,
            "world": {"width": 12, "height": 12},
            "stimuli": [{"type": "body"}, {"type": "food", "count": 40, "regrow_rate": 0.2}],
            "network": {"hidden_max": 4, "hidden_init": 2, "steps_per_tick": 4},
            "ecosystem": {"capacity": 30, "initial_population": 10},
        }
    )
    s = EcosystemSession(cfg, "eco")
    s.step(5)
    f = s.frame(0)
    assert f["population"] == len(f["agents"]) > 0
    a = s.agent(f["agents"][0]["i"])
    assert "traits" in a and a["id"] >= 0


def test_app_only_reads_inside_its_roots(runs_root, tmp_path):
    app = App([runs_root])
    with pytest.raises(PermissionError):
        app.series(str(tmp_path))
    with pytest.raises(PermissionError):
        app.new_session({"config": str(tmp_path / "x.yaml")})


def test_http_server_end_to_end(runs_root):
    app = App([runs_root])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return r.read()

    try:
        assert b"Spiking-Evo" in get("/")
        runs = json.loads(get("/api/runs"))
        req = urllib.request.Request(
            base + "/api/sim",
            data=json.dumps({"run": runs[0]["path"], "genomes": "checkpoint"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        sid = json.loads(urllib.request.urlopen(req).read())["sid"]
        frame = json.loads(get(f"/api/frame?sid={sid}&n=3&arena=0"))
        assert frame["tick"] == 3
        with pytest.raises(urllib.error.HTTPError) as err:
            get("/../pyproject.toml")
        assert err.value.code == 404
    finally:
        httpd.shutdown()
