"""Stage D3 analysis. Hypotheses and tests fixed before running (pre-registered):

For every run (one seed of one condition), from ecosystem.csv (rows every 250 ticks):
    feed_rate(window)   = feed events per 1000 adult-steps in that window
    delta_feed          = feed_rate(last 10 000 ticks) - feed_rate(first 10 000 ticks)
    juvenile_mortality  = juvenile deaths / births over the last 20 000 ticks
    population          = mean population over the last 20 000 ticks
Runs that go extinct are counted and excluded from the means.

H1 (care is selected): delta_feed is larger with care than in the control, in each world.
H2 (care pays): juvenile_mortality is lower and population higher with care, in each world.
H3 (harshness favours care): the care-control difference in delta_feed is larger in the
    harsh world than in the mild one.
Tests: one-sided permutation tests on seed-level values (20 000 permutations), alpha 0.05;
H3: permutation test on the difference of differences.

    python scripts/compare_eco.py RUNS_DIR
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
RNG = np.random.default_rng(0)
N_PERM = 20000


def run_metrics(path: Path) -> dict | None:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    tick = np.array([int(r["tick"]) for r in rows])
    pop = np.array([int(r["population"]) for r in rows])
    juv = np.array([int(r["juveniles"]) for r in rows])
    feeds = np.array([float(r["feed_events"] or 0) for r in rows])
    births = np.array([int(r["births"]) for r in rows])
    jdeaths = np.array([int(r["deaths_juvenile"]) for r in rows])
    last = tick[-1]
    extinct = pop[-1] == 0 or last < 50000

    def rate(mask):
        adult_steps = ((pop - juv)[mask] * 250).sum()
        return 1000 * feeds[mask].sum() / adult_steps if adult_steps else np.nan

    early, late = tick <= 10000, tick > last - 10000
    tail = tick > last - 20000
    return {
        "extinct": bool(extinct),
        "delta_feed": rate(late) - rate(early),
        "feed_late": rate(late),
        "juvenile_mortality": jdeaths[tail].sum() / max(births[tail].sum(), 1),
        "population": pop[tail].mean(),
    }


def perm_greater(a, b):
    """P(mean(a) - mean(b) >= observed) under random relabelling."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(N_PERM):
        p = RNG.permutation(pool)
        cnt += p[: len(a)].mean() - p[len(a) :].mean() >= obs
    return obs, cnt / N_PERM


data = {}
for exp in sorted(ROOT.glob("eco_d3_*")):
    vals = [m for f in sorted(exp.glob("*/ecosystem.csv")) if (m := run_metrics(f))]
    data[exp.name] = vals
    ext = sum(v["extinct"] for v in vals)
    ok = [v for v in vals if not v["extinct"]]
    if not ok:
        print(f"{exp.name}: {len(vals)} runs, all extinct")
        continue
    m = {
        k: np.nanmean([v[k] for v in ok])
        for k in ("delta_feed", "feed_late", "juvenile_mortality", "population")
    }
    print(
        f"{exp.name:24s} runs {len(vals):2d} extinct {ext:2d}  feed rate late {m['feed_late']:6.2f}"
        f"  delta feed {m['delta_feed']:+6.2f}  juvenile mortality {m['juvenile_mortality']:.2f}"
        f"  population {m['population']:6.1f}"
    )


def vals(name, key):
    return [v[key] for v in data.get(name, []) if not v["extinct"] and np.isfinite(v[key])]


print("\n=== Pre-registered tests (one-sided permutation, alpha 0.05)")
for world in ("mild", "harsh"):
    care, ctrl = f"eco_d3_{world}_care", f"eco_d3_{world}_control"
    d, p = perm_greater(vals(care, "delta_feed"), vals(ctrl, "delta_feed"))
    print(f"H1 {world:5s}: delta_feed care - control = {d:+.2f}, p = {p:.3f}")
    d, p = perm_greater(vals(ctrl, "juvenile_mortality"), vals(care, "juvenile_mortality"))
    print(f"H2 {world:5s}: juvenile mortality control - care = {d:+.3f}, p = {p:.3f}")
    d, p = perm_greater(vals(care, "population"), vals(ctrl, "population"))
    print(f"H2 {world:5s}: population care - control = {d:+.1f}, p = {p:.3f}")
hc, hk = vals("eco_d3_harsh_care", "delta_feed"), vals("eco_d3_harsh_control", "delta_feed")
mc, mk = vals("eco_d3_mild_care", "delta_feed"), vals("eco_d3_mild_control", "delta_feed")
if all([hc, hk, mc, mk]):
    obs = (np.mean(hc) - np.mean(hk)) - (np.mean(mc) - np.mean(mk))
    cnt = 0
    for _ in range(N_PERM):
        a, b = RNG.permutation(hc + hk), RNG.permutation(mc + mk)
        cnt += (a[: len(hc)].mean() - a[len(hc) :].mean()) - (
            b[: len(mc)].mean() - b[len(mc) :].mean()
        ) >= obs
    print(
        f"H3: (harsh care-control) - (mild care-control) in delta_feed = {obs:+.2f}, "
        f"p = {cnt / N_PERM:.3f}"
    )
