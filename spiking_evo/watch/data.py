"""Finding runs on disk and reading their CSV files for the dashboard."""

from __future__ import annotations

import csv
import math
from pathlib import Path

CSV_FILES = ("generations.csv", "ecosystem.csv", "stimuli.csv", "terms.csv", "phases.csv")


def default_roots(base: Path) -> list[Path]:
    """Folders that usually hold runs: runs/, runs-pilot/, runs-archive/*/x/runs."""
    roots = [base / "runs", base / "runs-pilot"]
    roots += sorted(p for p in (base / "runs-archive").glob("*/x/runs") if p.is_dir())
    return [r for r in roots if r.is_dir()]


def _seed_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


def root_label(root: Path) -> str:
    """Short name of a runs folder: "runs", "runs-pilot", "archivio 2026-11-16-e0"..."""
    if root.name == "runs" and root.parent.name == "x":
        return f"archivio {root.parent.parent.name}"
    return root.name


def find_runs(roots: list[Path]) -> list[dict]:
    """Every <root>/<experiment>/<seed>/ with generations.csv or ecosystem.csv."""
    runs = []
    for root in roots:
        label = root_label(root)
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            seeds = [p for p in exp_dir.iterdir() if p.is_dir()]
            for run in sorted(seeds, key=lambda p: _seed_key(p.name)):
                kind = (
                    "generations"
                    if (run / "generations.csv").exists()
                    else "ecosystem"
                    if (run / "ecosystem.csv").exists()
                    else None
                )
                if kind is None or not (run / "config.yaml").exists():
                    continue
                snaps = sorted(p.name for p in (run / "snapshots").glob("*.json"))
                runs.append(
                    {
                        "path": str(run.resolve()),
                        "root": str(root),
                        "source": label,
                        "experiment": exp_dir.name,
                        "seed": run.name,
                        "kind": kind,
                        "snapshots": snaps,
                        "checkpoint": (run / "checkpoint.json").exists(),
                    }
                )
    return runs


def _number(v: str) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_csv_columns(path: Path) -> dict[str, list]:
    """Columns of a CSV file as lists; numbers parsed, empty or broken cells become None.
    Rows cut short (a run killed while writing) are dropped."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = [r for r in reader if None not in r.values() and None not in r]
    out: dict[str, list] = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            x = _number(r[c])
            out[c].append(x if x is not None else (r[c] or None))
    return out


def run_series(run: Path) -> dict[str, dict[str, list]]:
    """All the CSV files of a run that exist, as columns."""
    return {name: read_csv_columns(run / name) for name in CSV_FILES if (run / name).exists()}
