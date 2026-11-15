"""Read ``runs/`` and summarise experiments across seeds: tables and plots.

    python -m spiking_evo analyze runs --out report

Writes ``report/summary.md`` (one table per experiment, mean ± std across seeds over the
last ``tail`` generations), ``report/summary.csv``, and PNG plots:

* ``fitness.png``: mean fitness per generation, one line per experiment, band = ±1 std
  across seeds;
* ``<experiment>_stimuli.png``: the per-stimulus counters of ``stimuli.csv``;
* ``<experiment>_seasons.png``: energy per living step and share of steps with an action,
  per season (only for experiments with a ``season`` stimulus).

Plots need matplotlib (``pip install -e ".[analysis]"``); without it only the tables are
written.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GEN_FIELDS = ("fitness_mean", "fitness_max", "alive_fraction", "energy_gained_mean",
              "firing_rate_mean", "hidden_mean", "synapses_mean")  # fmt: skip


@dataclass
class Experiment:
    name: str
    seeds: list[int] = field(default_factory=list)
    gen: dict[str, np.ndarray] = field(default_factory=dict)  # field -> (seeds, gens)
    stim: dict[str, np.ndarray] = field(default_factory=dict)  # stat -> (seeds, gens)

    @property
    def generations(self) -> int:
        return self.gen["fitness_mean"].shape[1] if self.gen else 0

    def season_names(self) -> list[str]:
        return sorted(
            {k.split(".", 1)[1][len("steps_"):] for k in self.stim
             if k.startswith("season.steps_")},
            key=lambda s: list(self.stim).index(f"season.steps_{s}"),
        )  # fmt: skip


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_runs(root: str | Path) -> dict[str, Experiment]:
    """Collect every ``<root>/<experiment>/<seed>/generations.csv``. Runs are truncated to
    the shortest one of their experiment so that seeds can be averaged."""
    root = Path(root)
    raw: dict[str, list[tuple[int, list[dict], list[dict]]]] = defaultdict(list)
    for gens_csv in sorted(root.glob("*/*/generations.csv")):
        seed_dir = gens_csv.parent
        try:
            seed = int(seed_dir.name)
        except ValueError:
            continue
        rows = _read(gens_csv)
        stim_csv = seed_dir / "stimuli.csv"
        stim_rows = _read(stim_csv) if stim_csv.exists() else []
        if rows:
            raw[seed_dir.parent.name].append((seed, rows, stim_rows))
    experiments = {}
    for name, runs in raw.items():
        n = min(len(rows) for _, rows, _ in runs)
        exp = Experiment(name, seeds=[s for s, _, _ in runs])
        for f in GEN_FIELDS:
            exp.gen[f] = np.array([[float(r[f]) for r in rows[:n]] for _, rows, _ in runs])
        stats: dict[str, np.ndarray] = {}
        for i, (_, _, srows) in enumerate(runs):
            for r in srows:
                g = int(r["generation"])
                if g < n:
                    arr = stats.setdefault(r["stat"], np.full((len(runs), n), np.nan))
                    arr[i, g] = float(r["mean"])
        exp.stim = stats
        experiments[name] = exp
    return experiments


def tail_stats(values: np.ndarray, tail: int) -> tuple[float, float]:
    """Mean over the last ``tail`` generations of each seed, then mean ± std across seeds."""
    per_seed = np.nanmean(values[:, -tail:], axis=1)
    return float(np.mean(per_seed)), float(np.std(per_seed))


def season_table(exp: Experiment, tail: int) -> list[tuple[str, float, float, float, float]]:
    """(season, energy per step mean, std, action share mean, std) over the last gens."""
    out = []
    for s in exp.season_names():
        steps = np.maximum(exp.stim[f"season.steps_{s}"], 1)
        e = tail_stats(exp.stim[f"season.energy_{s}"] / steps, tail)
        a = tail_stats(exp.stim.get(f"season.actions_{s}", np.zeros_like(steps)) / steps, tail)
        out.append((s, *e, *a))
    return out


def summarise(experiments: dict[str, Experiment], tail: int = 10) -> tuple[str, list[dict]]:
    lines = [f"# Riepilogo esperimenti\n\nMedia ± deviazione standard tra seed, calcolata "
             f"sulle ultime {tail} generazioni di ogni seed.\n"]  # fmt: skip
    rows = []
    for name, exp in sorted(experiments.items()):
        lines.append(f"\n## {name}\n\nSeed: {len(exp.seeds)} ({', '.join(map(str, exp.seeds))}),"
                     f" generazioni: {exp.generations}\n")  # fmt: skip
        lines.append("| misura | prime 10 gen | ultime gen |\n|---|---|---|")
        for key, values in [*((f, exp.gen[f]) for f in GEN_FIELDS), *sorted(exp.stim.items())]:
            if key.startswith("season."):
                continue
            first = np.nanmean(values[:, :10], axis=1)
            m, s = tail_stats(values, tail)
            fmt = "{:.1%}" if key == "alive_fraction" else "{:.2f}"
            lines.append(f"| {key} | {fmt.format(np.mean(first))} | "
                         f"{fmt.format(m)} ± {fmt.format(s)} |")  # fmt: skip
            rows.append(
                {
                    "experiment": name,
                    "measure": key,
                    "seeds": len(exp.seeds),
                    "first10_mean": float(np.mean(first)),
                    "last_mean": m,
                    "last_std": s,
                }
            )
        seasons = season_table(exp, tail)
        if seasons:
            lines.append("\n| stagione | energia per passo | quota di passi con azione |\n"
                         "|---|---|---|")  # fmt: skip
            for s, em, es, am, as_ in seasons:
                lines.append(f"| {s} | {em:.2f} ± {es:.2f} | {am:.1%} ± {as_:.1%} |")
                rows.append({"experiment": name, "measure": f"season.{s}.energy_per_step",
                             "seeds": len(exp.seeds), "first10_mean": np.nan,
                             "last_mean": em, "last_std": es})  # fmt: skip
                rows.append({"experiment": name, "measure": f"season.{s}.action_share",
                             "seeds": len(exp.seeds), "first10_mean": np.nan,
                             "last_mean": am, "last_std": as_})  # fmt: skip
    return "\n".join(lines) + "\n", rows


def plot(experiments: dict[str, Experiment], out: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, exp in sorted(experiments.items()):
        v = exp.gen["fitness_mean"]
        x = np.arange(v.shape[1])
        m, s = v.mean(axis=0), v.std(axis=0)
        ax.plot(x, m, label=f"{name} ({len(exp.seeds)} seed)")
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set(
        xlabel="generazione", ylabel="fitness media", title="Fitness media (banda: ±1 std tra seed)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    files.append(out / "fitness.png")
    fig.tight_layout()
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)

    for name, exp in sorted(experiments.items()):
        stats = {k: v for k, v in exp.stim.items() if not k.startswith("season.")}
        if stats:
            cols = 3
            rows_n = int(np.ceil(len(stats) / cols))
            fig, axes = plt.subplots(rows_n, cols, figsize=(12, 3 * rows_n), squeeze=False)
            for ax, (k, v) in zip(axes.ravel(), sorted(stats.items()), strict=False):
                x = np.arange(v.shape[1])
                m, s = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
                ax.plot(x, m)
                ax.fill_between(x, m - s, m + s, alpha=0.2)
                ax.set_title(k, fontsize=9)
                ax.grid(alpha=0.3)
            for ax in axes.ravel()[len(stats) :]:
                ax.axis("off")
            fig.suptitle(f"{name}: contatori per creatura per episodio (media tra seed)")
            fig.tight_layout()
            files.append(out / f"{name}_stimuli.png")
            fig.savefig(files[-1], dpi=110)
            plt.close(fig)
        seasons = exp.season_names()
        if seasons:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
            for s in seasons:
                steps = np.maximum(exp.stim[f"season.steps_{s}"], 1)
                for ax, key in ((a1, "energy"), (a2, "actions")):
                    v = exp.stim[f"season.{key}_{s}"] / steps
                    ax.plot(np.nanmean(v, axis=0), label=s)
            a1.set(title="energia per passo vissuto", xlabel="generazione")
            a2.set(title="quota di passi con azione", xlabel="generazione")
            for ax in (a1, a2):
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            fig.suptitle(f"{name}: comportamento per stagione (media tra seed)")
            fig.tight_layout()
            files.append(out / f"{name}_seasons.png")
            fig.savefig(files[-1], dpi=110)
            plt.close(fig)
    return files


def analyze(root: str | Path, out: str | Path, tail: int = 10) -> dict:
    experiments = load_runs(root)
    if not experiments:
        raise SystemExit(f"no runs found under {root}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    text, rows = summarise(experiments, tail)
    (out / "summary.md").write_text(text, encoding="utf-8")
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    images = plot(experiments, out)
    return {"experiments": sorted(experiments), "summary": out / "summary.md", "plots": images}
