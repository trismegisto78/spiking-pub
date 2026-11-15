"""Behavioural test of an evolved genome: what does it do, and which senses does it use?

    python -m spiking_evo behaviour runs/<exp>/<seed>/snapshots/gen_0299.json \\
        --config spiking_evo/experiments/<exp>.yaml --episodes 50

The genome is evaluated on ``episodes`` fresh episodes (never used during evolution) in
several conditions: ``normal``, and ``blind:<stimulus>`` for every stimulus, where the agent
perceives nothing from that stimulus while its effects on the world stay the same. If being
blind to a stimulus changes the outcome, the behaviour depends on that sense.

For each condition the report gives mean ± std over episodes of the energy gained, the
survival, and every counter the stimuli expose (``stimuli.csv`` names).
"""

from __future__ import annotations

import numpy as np

from .config.schema import Config
from .genetics.genome import Genome
from .genetics.population import Evolution

EVAL_KEY_BASE = 1_000_000  # generation keys >= this are never used by evolution


def behaviour(
    cfg: Config, genome: Genome, episodes: int = 50, conditions: list[str] | None = None
) -> dict[str, dict[str, tuple[float, float]]]:
    names = [s["name"] if "name" in s else s["type"] for s in cfg.stimuli]
    conditions = conditions or ["normal", *(f"blind:{n}" for n in names)]
    report: dict[str, dict[str, tuple[float, float]]] = {}
    for cond in conditions:
        blind = () if cond == "normal" else (cond.split(":", 1)[1],)
        evo = Evolution(cfg, blind=blind)
        g = genome.repair(evo.spec)
        rows: dict[str, list[float]] = {}
        for k in range(episodes):
            ev = evo.evaluate([g], EVAL_KEY_BASE + k)
            values = {
                "fitness": ev.fitness[0],
                "energy_gained": ev.stats.energy_gained[0],
                "alive_at_end": float(ev.stats.final_energy[0] > 0),
                "steps_alive": ev.stats.steps_alive[0],
                **{key: v[0] for key, v in ev.stimulus_stats.items()},
            }
            for key, v in values.items():
                rows.setdefault(key, []).append(float(v))
        report[cond] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in rows.items()}
    return report


def format_report(report: dict[str, dict[str, tuple[float, float]]]) -> str:
    conds = list(report)
    keys = list(report[conds[0]])
    head = "| misura | " + " | ".join(conds) + " |\n|---|" + "---|" * len(conds)
    lines = [head]
    for k in keys:
        cells = []
        for c in conds:
            m, s = report[c].get(k, (float("nan"), float("nan")))
            cells.append(f"{m:.1%} ± {s:.1%}" if k == "alive_at_end" else f"{m:.2f} ± {s:.2f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
