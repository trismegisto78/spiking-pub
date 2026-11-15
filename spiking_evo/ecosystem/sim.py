"""Stage D1: continuous evolution in one shared world (no generations, no fitness).

Agents live in a single arena with a fixed number of slots (``ecosystem.capacity``). They
are born, grow up, reproduce, age and die; selection is natural: what spreads is whatever
leaves living descendants.

Life cycle
    * birth: an adult with energy >= its ``repro_threshold`` (gene), at least
      ``repro_cooldown`` steps after its last reproduction, gives ``endowment`` (gene, share
      of its energy) to ``clutch`` (gene) offspring placed next to it. ``repro_overhead`` of
      that energy is lost. Each child inherits the parent's genome with mutations
      (``genetics.mutation``) and mutated life-history genes. No free slot, no birth.
    * juvenile (for ``juvenile_age`` steps, a gene): moves only on a share
      ``juvenile_speed`` of the steps, loses ``juvenile_damage`` times more energy to
      stimuli, may not use action outputs (unless ``juvenile_can_act``), cannot reproduce.
    * death: energy exhausted (``starvation``), or age >= ``max_age`` (``old age``).

Everything random derives from (seed, "eco", ...), so a run is reproducible.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config.loader import dump_config
from ..config.schema import Config
from ..core.clock import Clock
from ..core.world import World
from ..genetics.genome import Genome, GenomeSpec, build_network_batch
from ..genetics.operators import mutate
from ..genetics.population import n_input_channels
from ..metrics.logger import git_info
from ..seeding import derive_rng
from ..snn.controller import N_MOTOR_OUTPUTS, SNNController
from ..snn.encoders import make_encoder
from ..stimuli.registry import build_stimuli

LOG_COLUMNS = [
    "tick", "population", "juveniles", "births", "deaths_starvation", "deaths_old_age",
    "mean_age", "mean_age_at_death", "mean_energy", "max_generation", "lineages",
    "repro_threshold_mean", "endowment_mean", "clutch_mean", "juvenile_age_mean",
    "hidden_mean", "synapses_mean", "food_eaten", "hazard_steps",
    "awake_juveniles_mean", "awake_adults_mean", "dormant_left_mean", "growth_invested",
    "feed_events", "energy_given", "elapsed_s",
]  # fmt: skip


@dataclass
class Traits:
    """Life-history genes, one row per slot."""

    repro_threshold: np.ndarray
    endowment: np.ndarray
    clutch: np.ndarray
    juvenile_age: np.ndarray

    @classmethod
    def initial(cls, n: int, eco) -> Traits:
        return cls(
            repro_threshold=np.full(n, eco.init_repro_threshold),
            endowment=np.full(n, eco.init_endowment),
            clutch=np.full(n, eco.init_clutch, dtype=np.int64),
            juvenile_age=np.full(n, eco.init_juvenile_age),
        )

    def row(self, i: int) -> dict:
        return {
            "repro_threshold": float(self.repro_threshold[i]),
            "endowment": float(self.endowment[i]),
            "clutch": int(self.clutch[i]),
            "juvenile_age": float(self.juvenile_age[i]),
        }


class _LifeStageController:
    """Wraps the SNN controller: juveniles move slowly and (by default) cannot act."""

    def __init__(self, sim: Ecosystem) -> None:
        self.sim = sim

    def reset(self) -> None:
        self.sim.controller.reset()

    def act(self, obs, active):
        out = self.sim.controller.act(obs, active)
        juv = self.sim.world.agents.stage == 1
        forward, turn = out[0].copy(), out[1]
        eco = self.sim.cfg.ecosystem
        if eco.juvenile_speed < 1.0:
            period = max(1, round(1.0 / max(eco.juvenile_speed, 1e-9)))
            slow = juv & (self.sim.world.agents.age % period != 0)
            forward[slow] = 0
        if len(out) > 2:
            actions = out[2].copy()
            if not eco.juvenile_can_act:
                actions[juv] = False
            return forward, turn, actions
        return forward, turn

    def feedback(self, energy_delta, alive, stimulus_delta=None):
        self.sim.controller.feedback(energy_delta, alive, stimulus_delta)

    def energy_cost(self):
        return self.sim.controller.energy_cost()


class Ecosystem:
    def __init__(self, cfg: Config, out_dir: str | Path | None = None) -> None:
        if cfg.world.arenas != "shared":
            cfg.world.arenas = "shared"  # one world for everybody
        self.cfg = cfg
        eco = cfg.ecosystem
        self.K = eco.capacity
        n_in, self.input_names = n_input_channels(cfg)
        self.spec = GenomeSpec.from_config(cfg, n_in, N_MOTOR_OUTPUTS + cfg.network.action_outputs)
        self.rng = derive_rng(cfg.seed, "eco", "life")
        self.world = World(
            cfg.world,
            cfg.agent,
            build_stimuli(cfg.stimuli),
            n_agents=self.K,
            clock=Clock(cfg.network.dt_ms, cfg.network.steps_per_tick),
            n_actions=cfg.network.action_outputs,
        )
        self.out_dir = Path(out_dir) if out_dir else None
        self._init_population()

    # -- setup ----------------------------------------------------------------------
    def _init_population(self) -> None:
        cfg, eco, K = self.cfg, self.cfg.ecosystem, self.K
        self.world.reset(cfg.seed, "eco", "world")
        a = self.world.agents
        n0 = min(eco.initial_population, K)
        rng0 = derive_rng(cfg.seed, "eco", "init")
        if eco.init_from:
            from ..genetics.genome import adapt_genome
            from ..genetics.population import load_population

            source, names = load_population(eco.init_from)
            pick = rng0.choice(len(source), size=K, replace=len(source) < K)
            wire = derive_rng(cfg.seed, "eco", "adapt")
            n_out = self.spec.n_outputs
            self.genomes = [
                adapt_genome(source[i], names, self.input_names, n_out, self.spec, wire)
                for i in pick
            ]
        else:
            self.genomes = [Genome.random(self.spec, rng0) for _ in range(K)]
        a.alive[n0:] = False
        a.x[:n0] = rng0.integers(self.world.width, size=n0)
        a.y[:n0] = rng0.integers(self.world.height, size=n0)
        a.heading[:n0] = rng0.integers(4, size=n0)
        a.age[:n0] = rng0.integers(0, int(eco.init_juvenile_age) + 1, size=n0)  # mixed ages
        self.traits = Traits.initial(K, eco)
        self.ids = np.full(K, -1, dtype=np.int64)
        self.ids[:n0] = np.arange(n0)
        self.next_id = n0
        self.parent = np.full(K, -1, dtype=np.int64)
        self.lineage = self.ids.copy()
        self.generation = np.zeros(K, dtype=np.int64)
        self.last_birth = np.full(K, -(10**9), dtype=np.int64)
        # Ids and parents, shared with stimuli (kin) through world.extra.
        self.world.extra["id"] = self.ids
        self.world.extra["parent"] = self.parent
        if cfg.network.development.enabled:
            dev = cfg.network.development
            rng_d = derive_rng(cfg.seed, "eco", "dormant")
            for g in self.genomes:
                if not g.dormant.any():  # e.g. a seed species evolved without development
                    hidden = np.arange(g.n_neurons) >= self.spec.n_fixed
                    g.dormant = hidden & (rng_d.random(g.n_neurons) < dev.init_dormant_prob)
                    g.wake_at = np.full(g.n_neurons, dev.init_wake_at)
        self.tick = 0
        self.encoder = make_encoder(
            cfg.network.encoder,
            max_rate_hz=cfg.network.max_rate_hz,
            dt_ms=cfg.network.dt_ms,
            current_gain=cfg.network.current_gain,
        )
        net = build_network_batch(self.genomes, self.spec)
        self.controller = SNNController(
            net,
            self.encoder,
            dt_ms=cfg.network.dt_ms,
            steps_per_tick=cfg.network.steps_per_tick,
            bias_input=cfg.network.bias_input,
            motor_threshold_hz=cfg.network.motor_threshold_hz,
            rng=derive_rng(cfg.seed, "eco", "encoder"),
            action_threshold_hz=cfg.network.action_threshold_hz,
            plasticity=cfg.network.plasticity if cfg.network.plasticity.enabled else None,
            weight_max=cfg.network.weight_max,
            development=cfg.network.development if cfg.network.development.enabled else None,
        )
        self.controller.reset()
        self.wrapper = _LifeStageController(self)
        self._window = self._empty_window()
        self._update_stage()

    def _empty_window(self) -> dict:
        return {"births": 0, "deaths_starvation": 0, "deaths_old_age": 0, "death_ages": []}

    # -- one step -----------------------------------------------------------------------
    def _update_stage(self) -> None:
        a = self.world.agents
        a.stage[:] = (a.alive & (a.age < self.traits.juvenile_age)).astype(np.int64)

    def step(self) -> None:
        eco = self.cfg.ecosystem
        a = self.world.agents
        alive_before = a.alive.copy()
        gained_before = a.energy_gained.copy()
        self.world.step(self.wrapper)
        # Juveniles are fragile: they lose more to stimuli.
        if eco.juvenile_damage != 1.0:
            lost = np.maximum(gained_before - a.energy_gained, 0.0)
            extra = (eco.juvenile_damage - 1.0) * lost * (a.stage == 1) * alive_before
            a.energy -= extra
            a.energy_gained -= extra
            starving = a.alive & (a.energy <= 0)
            a.alive &= ~starving
            np.maximum(a.energy, 0.0, out=a.energy)
        old = a.alive & (a.age >= eco.max_age)
        a.alive &= ~old
        died = alive_before & ~a.alive
        w = self._window
        w["deaths_old_age"] += int(old.sum())
        w["deaths_starvation"] += int((died & ~old).sum())
        w["death_ages"].extend(a.age[died].tolist())
        self._grow()
        self._reproduce()
        self.tick += 1
        self._update_stage()

    def _grow(self) -> None:
        """Development by growth: agents with dormant neurons invest energy in their brain."""
        dev = self.cfg.network.development
        if not (dev.enabled and dev.trigger == "growth"):
            return
        eco, a = self.cfg.ecosystem, self.world.agents
        left = self.controller.dormant_left() > 0
        invest = np.where(
            a.alive & left,
            np.minimum(eco.growth_rate, np.maximum(a.energy - eco.growth_reserve, 0)),
            0.0,
        )
        a.energy -= invest
        self._window["growth"] = self._window.get("growth", 0.0) + float(invest.sum())
        self.controller.invest_in_growth(invest)

    def _reproduce(self) -> None:
        eco, a, tr = self.cfg.ecosystem, self.world.agents, self.traits
        ready = (
            a.alive
            & (a.stage == 0)
            & (a.energy >= tr.repro_threshold)
            & (self.tick - self.last_birth >= eco.repro_cooldown)
        )
        parents = np.flatnonzero(ready)
        if parents.size == 0:
            return
        free = list(np.flatnonzero(~a.alive))
        if not free:
            return
        parents = self.rng.permutation(parents)
        born_slots, newborn = [], np.zeros(self.K, dtype=bool)
        for p in parents:
            if not free:
                break
            n = int(min(tr.clutch[p], len(free)))
            budget = a.energy[p] * tr.endowment[p]
            a.energy[p] -= budget
            self.last_birth[p] = self.tick
            each = budget * (1.0 - eco.repro_overhead) / n
            r = eco.spawn_radius
            for _ in range(n):
                c = free.pop(0)
                dx, dy = self.rng.integers(-r, r + 1, size=2)
                self.world.spawn(
                    np.array([c]),
                    np.array([a.x[p] + dx]),
                    np.array([a.y[p] + dy]),
                    np.array([self.rng.integers(4)]),
                    np.array([each]),
                    stage=1,
                )
                self.genomes[c] = mutate(self.genomes[p], self.spec, self.cfg.genetics.mutation,
                                         self.rng)  # fmt: skip
                self._inherit_traits(p, c)
                self.ids[c] = self.next_id
                self.next_id += 1
                self.parent[c] = self.ids[p]
                self.lineage[c] = self.lineage[p]
                self.generation[c] = self.generation[p] + 1
                self.last_birth[c] = self.tick
                born_slots.append(c)
                newborn[c] = True
        if born_slots:
            self._window["births"] += len(born_slots)
            net = build_network_batch(self.genomes, self.spec)
            self.controller.replace_networks(net, keep=~newborn)

    def _inherit_traits(self, p: int, c: int) -> None:
        eco, tr, rng = self.cfg.ecosystem, self.traits, self.rng
        s = eco.trait_sigma
        tr.repro_threshold[c] = np.clip(
            tr.repro_threshold[p] * np.exp(rng.normal(0, s)), *eco.repro_threshold_range
        )
        tr.endowment[c] = np.clip(tr.endowment[p] * np.exp(rng.normal(0, s)), *eco.endowment_range)
        tr.juvenile_age[c] = np.clip(
            max(tr.juvenile_age[p], 1.0) * np.exp(rng.normal(0, s)), *eco.juvenile_age_range
        )
        clutch = tr.clutch[p]
        if rng.random() < eco.clutch_change:
            clutch += rng.choice(np.array([-1, 1]))
        tr.clutch[c] = int(np.clip(clutch, 1, eco.clutch_max))

    # -- run ------------------------------------------------------------------------------
    def stats(self, elapsed: float) -> dict:
        a, tr, w = self.world.agents, self.traits, self._window
        alive = a.alive
        n = int(alive.sum())

        def mean(x):
            return float(np.mean(x[alive])) if n else 0.0

        stim = {s.name: s for s in self.world.stimuli}
        food = stim.get("food")
        kin = stim.get("kin")
        dev = self.cfg.network.development.enabled
        awake = self.controller.network.awake_count(self.spec.n_inputs) if dev else np.zeros(self.K)
        juv = alive & (a.stage == 1)
        adu = alive & (a.stage == 0)
        if kin is not None:
            fe, eg = float(kin.feed_events.sum()), float(kin.energy_given.sum())
            prev = getattr(self, "_kin_prev", (0.0, 0.0))
            self._kin_prev = (fe, eg)
            feed_events, energy_given = fe - prev[0], eg - prev[1]
        else:
            feed_events = energy_given = ""
        hazard = stim.get("hazard")
        row = {
            "tick": self.tick,
            "population": n,
            "juveniles": int((a.stage == 1).sum()),
            "births": w["births"],
            "deaths_starvation": w["deaths_starvation"],
            "deaths_old_age": w["deaths_old_age"],
            "mean_age": mean(a.age),
            "mean_age_at_death": float(np.mean(w["death_ages"])) if w["death_ages"] else "",
            "mean_energy": mean(a.energy),
            "max_generation": int(self.generation[alive].max()) if n else 0,
            "lineages": len(set(self.lineage[alive].tolist())),
            "repro_threshold_mean": mean(tr.repro_threshold),
            "endowment_mean": mean(tr.endowment),
            "clutch_mean": mean(tr.clutch.astype(float)),
            "juvenile_age_mean": mean(tr.juvenile_age),
            "hidden_mean": float(np.mean([self.genomes[i].n_hidden for i in np.flatnonzero(alive)]))
            if n
            else 0.0,
            "synapses_mean": float(
                np.mean([self.genomes[i].n_synapses for i in np.flatnonzero(alive)])
            )
            if n
            else 0.0,
            "food_eaten": int(food.eaten.sum()) if food is not None else "",
            "hazard_steps": int(hazard.steps_inside.sum()) if hazard is not None else "",
            "awake_juveniles_mean": float(awake[juv].mean()) if juv.any() and dev else "",
            "awake_adults_mean": float(awake[adu].mean()) if adu.any() and dev else "",
            "dormant_left_mean": float(self.controller.dormant_left()[alive].mean())
            if n and dev
            else "",
            "growth_invested": round(w.get("growth", 0.0), 2),
            "feed_events": feed_events,
            "energy_given": energy_given,
            "elapsed_s": round(elapsed, 2),
        }
        return row

    def snapshot(self) -> dict:
        a = self.world.agents
        alive = np.flatnonzero(a.alive)
        return {
            "format": "spiking_evo.ecosystem_snapshot",
            "version": 1,
            "tick": self.tick,
            "input_names": self.input_names,
            "agents": [
                {
                    "id": int(self.ids[i]),
                    "parent": int(self.parent[i]),
                    "lineage": int(self.lineage[i]),
                    "generation": int(self.generation[i]),
                    "age": int(a.age[i]),
                    "energy": float(a.energy[i]),
                    "traits": self.traits.row(i),
                    "genome": self.genomes[i].to_dict(),
                }
                for i in alive
            ],
        }

    def run(self, ticks: int | None = None, progress=None) -> list[dict]:
        eco = self.cfg.ecosystem
        ticks = ticks if ticks is not None else eco.ticks
        rows: list[dict] = []
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "snapshots").mkdir(exist_ok=True)
            extra = {"_run": {**git_info(Path(__file__).resolve().parent), "mode": "ecosystem"}}
            (self.out_dir / "config.yaml").write_text(dump_config(self.cfg, extra))
            with (self.out_dir / "ecosystem.csv").open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()
        t0 = time.perf_counter()
        for _ in range(ticks):
            self.step()
            if self.tick % eco.log_every == 0:
                row = self.stats(time.perf_counter() - t0)
                rows.append(row)
                self._window = self._empty_window()
                if self.out_dir:
                    with (self.out_dir / "ecosystem.csv").open("a", newline="") as f:
                        csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)
                if progress:
                    progress(row)
            if self.out_dir and eco.snapshot_every and self.tick % eco.snapshot_every == 0:
                self._write_snapshot()
            if not self.world.agents.alive.any():
                break  # extinction
        if self.out_dir:
            self._write_snapshot()
        return rows

    def _write_snapshot(self) -> None:
        path = self.out_dir / "snapshots" / f"tick_{self.tick:07d}.json"
        path.write_text(json.dumps(self.snapshot()))
