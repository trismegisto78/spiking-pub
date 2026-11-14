"""Per-generation metrics, snapshots and checkpoints on disk.

Layout of ``runs/<experiment>/<seed>/``:

    config.yaml               config used, plus repository commit hash
    generations.csv           one row per generation (fitness stats, activity, size)
    terms.csv                 mean / min / max contribution of every fitness term
    snapshots/gen_XXXX.json   best genome every ``snapshot_every`` generations
    checkpoint.json           whole population, to resume the run
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..config.loader import dump_config
from ..config.schema import Config

GENERATION_COLUMNS = [
    "generation",
    "fitness_mean",
    "fitness_max",
    "fitness_min",
    "fitness_var",
    "n_species",
    "firing_rate_mean",
    "energy_mean",
    "energy_gained_mean",
    "energy_gained_max",
    "alive_fraction",
    "visited_cells_mean",
    "hidden_mean",
    "synapses_mean",
    "best_hidden",
    "best_synapses",
    "elapsed_s",
]


def git_info(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def run_dir(cfg: Config, out_root: str | Path | None = None) -> Path:
    return Path(out_root or cfg.metrics.out_dir) / cfg.experiment / str(cfg.seed)


class RunLogger:
    def __init__(self, directory: Path, cfg: Config, term_names: tuple[str, ...]) -> None:
        self.dir = Path(directory)
        self.cfg = cfg
        self.term_columns = ["generation"] + [
            f"{t}_{stat}" for t in term_names for stat in ("mean", "min", "max")
        ]
        self.term_names = term_names

    @property
    def generations_csv(self) -> Path:
        return self.dir / "generations.csv"

    @property
    def terms_csv(self) -> Path:
        return self.dir / "terms.csv"

    @property
    def checkpoint_path(self) -> Path:
        return self.dir / "checkpoint.json"

    def start(self, first_generation: int = 0) -> None:
        """Create the run directory. When resuming, rows from ``first_generation`` on are
        dropped so the CSV files stay consistent with the checkpoint."""
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        repo = Path(__file__).resolve().parent
        extra = {"_run": {**git_info(repo), "resumed_from_generation": first_generation or None}}
        (self.dir / "config.yaml").write_text(dump_config(self.cfg, extra), encoding="utf-8")
        for path, cols in (
            (self.generations_csv, GENERATION_COLUMNS),
            (self.terms_csv, self.term_columns),
        ):
            rows = []
            if first_generation > 0 and path.exists():
                with path.open(newline="") as f:
                    rows = [r for r in csv.DictReader(f) if int(r["generation"]) < first_generation]
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)

    def log_generation(self, row: dict[str, Any], terms: dict[str, Any]) -> None:
        self._append(self.generations_csv, GENERATION_COLUMNS, row)
        self._append(self.terms_csv, self.term_columns, terms)

    def snapshot(self, generation: int, genome: dict[str, Any], fitness: float) -> Path:
        path = self.dir / "snapshots" / f"gen_{generation:04d}.json"
        payload = {"generation": generation, "fitness": fitness, "genome": genome}
        _atomic_write(path, json.dumps(payload))
        return path

    def checkpoint(self, next_generation: int, genomes: list[dict[str, Any]]) -> None:
        payload = {
            "format": "spiking_evo.checkpoint",
            "version": 1,
            "experiment": self.cfg.experiment,
            "seed": self.cfg.seed,
            "next_generation": next_generation,
            "genomes": genomes,
        }
        _atomic_write(self.checkpoint_path, json.dumps(payload))

    def load_checkpoint(self) -> dict[str, Any]:
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if data.get("format") != "spiking_evo.checkpoint" or data.get("version") != 1:
            raise ValueError("unsupported checkpoint format")
        if data["seed"] != self.cfg.seed or data["experiment"] != self.cfg.experiment:
            raise ValueError("checkpoint belongs to a different experiment or seed")
        return data

    @staticmethod
    def _append(path: Path, cols: list[str], row: dict[str, Any]) -> None:
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writerow(row)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
