"""Stage D6 analysis: D5 with a gift that matters (feed_amount 10 instead of 2).

Hypotheses and tests fixed before running (pre-registered, 16/11/2026), identical to D5:

Two conditions, seeds 1-16 each, 200 000 ticks: eco_d6_care (feeding transfers 10 energy,
8 received) and eco_d6_control (feed_amount 0: the action exists but does nothing). Run on
the notebook in two batches of 8+8 seeds (1-8, then 9-16). The confirmatory analysis is the
full 16+16; the first batch (8+8) is an interim look, reported as descriptive only.
Per run, from ecosystem.csv:
    feed_rate(window)  = feed events per 1000 adult-steps in that window
    feed_late          = feed_rate(last 20 000 ticks)
    delta_feed         = feed_late - feed_rate(first 20 000 ticks)
    juvenile_mortality = juvenile deaths / births over the last 20 000 ticks
    dormant_birth      = mean dormant neurons at birth over the last 20 000 ticks
Extinctions are an outcome: counted per condition, runs excluded from the other means.

H1 (care is selected): feed_late is higher with care than in the control.
H2 (care spreads): delta_feed is higher with care than in the control.
H3 (care pays): juvenile_mortality is lower with care.
Exploratory, no test: extinctions per condition; dormant_birth and population, care vs
control.
Tests: one-sided permutation tests on seed-level values (20 000 permutations), alpha 0.05.

    python scripts/compare_d6.py RUNS_DIR [MIN_TICK]
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
RNG = np.random.default_rng(0)
N_PERM = 20000
WIN = 20000
# Runs stopped early (budget) count as surviving if they reached MIN_TICK; extinct runs
# end at their last log, far below it.
MIN_TICK = int(sys.argv[2]) if len(sys.argv) > 2 else 190000


def run_metrics(path: Path) -> dict | None:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {"extinct": True}
    col = lambda k: np.array([float(r.get(k) or 0) for r in rows])  # noqa: E731
    tick, pop, juv = col("tick"), col("population"), col("juveniles")
    feeds, births, jd = col("feed_events"), col("births"), col("deaths_juvenile")
    step = tick[0] if len(tick) == 1 else tick[1] - tick[0]
    last = tick[-1]
    if pop[-1] == 0 or last < MIN_TICK:
        return {"extinct": True}

    def rate(mask):
        adult = ((pop - juv)[mask] * step).sum()
        return 1000 * feeds[mask].sum() / adult if adult else np.nan

    early, late = tick <= WIN, tick > last - WIN
    return {
        "extinct": False,
        "feed_late": rate(late),
        "delta_feed": rate(late) - rate(early),
        "juvenile_mortality": jd[late].sum() / max(births[late].sum(), 1),
        "dormant_birth": col("dormant_at_birth_mean")[late].mean(),
        "population": pop[late].mean(),
    }


def perm_greater(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.nan, np.nan
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    cnt = sum(
        (p := RNG.permutation(pool))[: len(a)].mean() - p[len(a) :].mean() >= obs
        for _ in range(N_PERM)
    )
    return obs, cnt / N_PERM


data = {}
for name in ("eco_d6_care", "eco_d6_control"):
    runs = [run_metrics(f) for f in sorted((ROOT / name).glob("*/ecosystem.csv"))]
    data[name] = [r for r in runs if not r["extinct"]]
    ext = sum(r["extinct"] for r in runs)
    print(f"{name:16s} runs {len(runs):2d} extinct {ext:2d}", end="")
    for k in ("feed_late", "delta_feed", "juvenile_mortality", "dormant_birth", "population"):
        v = [r[k] for r in data[name] if np.isfinite(r[k])]
        print(f"  {k} {np.mean(v) if v else np.nan:.3f}", end="")
    print()
    for f, r in zip(sorted((ROOT / name).glob("*/ecosystem.csv")), runs, strict=True):
        if not r["extinct"]:
            vals = " ".join(f"{k}={r[k]:.3f}" for k in r if k != "extinct")
            print(f"   {f.parent.name:>3s} {vals}")


def v(name, key):
    return [r[key] for r in data[name] if np.isfinite(r[key])]


print("\n=== Pre-registered tests (one-sided permutation, alpha 0.05)")
c, k = "eco_d6_care", "eco_d6_control"
d, p = perm_greater(v(c, "feed_late"), v(k, "feed_late"))
print(f"H1 feed_late care - control = {d:+.3f}, p = {p:.3f}")
d, p = perm_greater(v(c, "delta_feed"), v(k, "delta_feed"))
print(f"H2 delta_feed care - control = {d:+.3f}, p = {p:.3f}")
d, p = perm_greater(v(k, "juvenile_mortality"), v(c, "juvenile_mortality"))
print(f"H3 juvenile mortality control - care = {d:+.3f}, p = {p:.3f}")
