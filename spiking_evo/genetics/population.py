"""Generational loop: evaluate -> log -> select -> reproduce.

Randomness is keyed by generation, so generation g is reproducible in isolation:

    initial population      derive_rng(seed, "init")
    evaluation episode e    world/stimuli: (seed, "eval", g, "episode", e, ...)
                            encoder:       (seed, "eval", g, "episode", e, "encoder")
    reproduction            derive_rng(seed, "reproduce", g)

With ``workers > 1`` the population is split into contiguous chunks evaluated by a pool of
processes. Chunks draw their random numbers as slices of whole-population draws (see
``World.per_arena``), so the results are identical for any number of workers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config.loader import phase_configs
from ..config.schema import Config
from ..core.clock import Clock
from ..core.world import World
from ..fitness.evaluator import TERM_NAMES, BehaviourStats, compute_terms, total_fitness
from ..metrics.logger import RunLogger, run_dir
from ..seeding import derive_rng
from ..snn.controller import N_MOTOR_OUTPUTS, SNNController
from ..snn.encoders import make_encoder
from ..stimuli.registry import build_stimuli
from .genome import Genome, GenomeSpec, adapt_genome, build_network_batch
from .operators import crossover, mutate, tournament_select


@dataclass
class Evaluation:
    fitness: np.ndarray
    terms: dict[str, np.ndarray]
    stats: BehaviourStats
    # Per-agent counters reported by stimuli (``episode_stats``), keyed "<stimulus>.<stat>".
    stimulus_stats: dict[str, np.ndarray] = field(default_factory=dict)


def n_input_channels(cfg: Config) -> tuple[int, list[str]]:
    """Input size of the network, derived from the channels declared by the stimuli."""
    stimuli = build_stimuli(cfg.stimuli)
    names = [f"{s.name}.{c}" for s in stimuli for c in s.channels]
    if cfg.network.bias_input:
        names.append("bias")
    return len(names), names


class Evolution:
    def __init__(self, cfg: Config, workers: int = 1, blind: tuple[str, ...] = ()) -> None:
        """``blind``: names of stimuli whose sensory channels read as zero (for ablation
        tests); their effects on the world are unchanged."""
        self.base_cfg = cfg
        self.blind = tuple(blind)
        self.phases = phase_configs(cfg)
        self.phase_index = -1
        self._activate_phase(0)
        if workers > 1 and cfg.world.arenas != "isolated":
            raise ValueError("parallel evaluation needs world.arenas: isolated")
        self.workers = max(1, min(workers, cfg.genetics.population))
        self._pool: ProcessPoolExecutor | None = None

    # -- phases ----------------------------------------------------------------------
    def phase_at(self, generation: int) -> int:
        return max(i for i, (start, _, _) in enumerate(self.phases) if start <= generation)

    def _activate_phase(self, index: int) -> bool:
        """Switch config, senses and genome layout to phase ``index``; True if changed."""
        if index == self.phase_index:
            return False
        self.phase_index = index
        self.phase_name = self.phases[index][1]
        cfg = self.phases[index][2]
        self.cfg = cfg
        n_in, self.input_names = n_input_channels(cfg)
        self.spec = GenomeSpec.from_config(cfg, n_in, N_MOTOR_OUTPUTS + cfg.network.action_outputs)
        if getattr(self, "_pool", None) is not None:
            self.close()  # workers hold the previous phase's config
        return True

    def adapt(self, genomes: list[Genome], old_inputs: list[str]) -> list[Genome]:
        """Carry genomes from another sensory layout over to the current phase."""
        n_out = self.spec.n_outputs
        if list(old_inputs) == self.input_names and all(g.n_outputs == n_out for g in genomes):
            return [g.repair(self.spec) for g in genomes]
        return [adapt_genome(g, old_inputs, self.input_names, n_out, self.spec) for g in genomes]

    # -- population ---------------------------------------------------------------
    def initial_population(self) -> list[Genome]:
        rng = derive_rng(self.cfg.seed, "init")
        return [Genome.random(self.spec, rng) for _ in range(self.cfg.genetics.population)]

    def evaluate(self, genomes: list[Genome], generation: int) -> Evaluation:
        if self.workers == 1:
            return self.evaluate_chunk(genomes, generation, 0, len(genomes))
        pool = self._get_pool()
        bounds = np.linspace(0, len(genomes), self.workers + 1).astype(int)
        futures = [
            pool.submit(_evaluate_in_worker, genomes[lo:hi], generation, int(lo), len(genomes))
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
            if hi > lo
        ]
        return merge_evaluations([f.result() for f in futures])

    def evaluate_chunk(
        self, genomes: list[Genome], generation: int, offset: int, total: int
    ) -> Evaluation:
        """Evaluate agents [offset, offset + len(genomes)) of a population of ``total``."""
        cfg, seed = self.cfg, self.cfg.seed
        net = build_network_batch(genomes, self.spec)
        world = World(
            cfg.world,
            cfg.agent,
            [Blinded(s) if s.name in self.blind else s for s in build_stimuli(cfg.stimuli)],
            n_agents=len(genomes),
            clock=Clock(cfg.network.dt_ms, cfg.network.steps_per_tick),
            batch_offset=offset,
            batch_total=total,
            n_actions=cfg.network.action_outputs,
        )
        encoder = make_encoder(
            cfg.network.encoder,
            max_rate_hz=cfg.network.max_rate_hz,
            dt_ms=cfg.network.dt_ms,
            current_gain=cfg.network.current_gain,
        )
        per_episode = []
        stim_episodes: list[dict[str, np.ndarray]] = []
        for ep in range(cfg.fitness.episodes):
            keys = ("eval", generation, "episode", ep)
            controller = SNNController(
                net,
                encoder,
                dt_ms=cfg.network.dt_ms,
                steps_per_tick=cfg.network.steps_per_tick,
                bias_input=cfg.network.bias_input,
                motor_threshold_hz=cfg.network.motor_threshold_hz,
                rng=derive_rng(seed, *keys, "encoder"),
                action_threshold_hz=cfg.network.action_threshold_hz,
                batch_offset=offset,
                batch_total=total,
            )
            res = world.run_episode(controller, seed, *keys)
            stim_episodes.append(
                {
                    f"{s.name}.{k}": np.asarray(v, dtype=np.float64)
                    for s in world.stimuli
                    if hasattr(s, "episode_stats")
                    for k, v in s.episode_stats().items()
                }
            )
            per_episode.append(
                BehaviourStats(
                    energy_gained=res.energy_gained,
                    final_energy=res.final_energy,
                    firing_rate_hz=controller.mean_firing_rate_hz(),
                    noninput_spikes=controller.noninput_spikes.copy(),
                    visited_cells=res.visited_cells.astype(np.float64),
                    steps_alive=res.steps_alive.astype(np.float64),
                )
            )
        terms_list = [compute_terms(cfg.fitness, s) for s in per_episode]
        terms = {t: np.mean([tl[t] for tl in terms_list], axis=0) for t in TERM_NAMES}
        stats = BehaviourStats(
            **{
                f: np.mean([getattr(s, f) for s in per_episode], axis=0)
                for f in BehaviourStats.__dataclass_fields__
            }
        )
        stim = {k: np.mean([e[k] for e in stim_episodes], axis=0) for k in stim_episodes[0]}
        return Evaluation(
            fitness=total_fitness(terms), terms=terms, stats=stats, stimulus_stats=stim
        )

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            limit_native_threads()  # inherited by the workers
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.cfg, self.blind),
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def reproduce(
        self, genomes: list[Genome], fitness: np.ndarray, generation: int
    ) -> list[Genome]:
        g_cfg = self.cfg.genetics
        rng = derive_rng(self.cfg.seed, "reproduce", generation)
        order = np.argsort(-fitness, kind="stable")
        nxt = [genomes[i].copy() for i in order[: g_cfg.elitism]]
        n_children = len(genomes) - len(nxt)
        parents = tournament_select(fitness, g_cfg.tournament_size, n_children, rng)
        for p in parents:
            if g_cfg.crossover > 0 and rng.random() < g_cfg.crossover:
                q = tournament_select(fitness, g_cfg.tournament_size, 1, rng)[0]
                a, b = (p, q) if fitness[p] >= fitness[q] else (q, p)
                child = crossover(genomes[a], genomes[b], self.spec, rng)
            else:
                child = genomes[p]
            nxt.append(mutate(child, self.spec, g_cfg.mutation, rng))
        return nxt

    # -- main loop -----------------------------------------------------------------
    def run(
        self,
        out_root: str | Path | None = None,
        resume: bool = False,
        progress: Callable[[dict], None] | None = None,
        init_from: str | Path | None = None,
    ) -> list[Genome]:
        """Run the generational loop.

        ``init_from``: a run directory (or its checkpoint.json) of another experiment; its
        final population replaces the random initial one, adapted to this run's senses.
        """
        cfg = self.base_cfg
        logger = RunLogger(run_dir(cfg, out_root), cfg, TERM_NAMES)
        start = 0
        if resume and logger.checkpoint_path.exists():
            ckpt = logger.load_checkpoint()
            start = int(ckpt["next_generation"])
            self._activate_phase(self.phase_at(start))
            population = [Genome.from_dict(d) for d in ckpt["genomes"]]
            population = self.adapt(population, ckpt.get("input_names") or self.input_names)
        elif init_from is not None:
            self._activate_phase(0)
            genomes, names = load_population(init_from)
            population = self.adapt(genomes, names)
            if len(population) != cfg.genetics.population:
                raise ValueError(
                    f"init_from has {len(population)} genomes, the config wants "
                    f"{cfg.genetics.population}"
                )
        else:
            population = self.initial_population()
        logger.start(start)
        logger.log_phase(start, self.phase_name, self.input_names)
        last = cfg.genetics.generations - 1
        evaluated = population
        try:
            for gen in range(start, cfg.genetics.generations):
                old_inputs = self.input_names
                if self._activate_phase(self.phase_at(gen)):
                    population = self.adapt(population, old_inputs)
                    logger.log_phase(gen, self.phase_name, self.input_names)
                cfg = self.cfg
                t0 = time.perf_counter()
                ev = self.evaluate(population, gen)
                best = int(np.argmax(ev.fitness))
                row = self._generation_row(gen, population, ev, best, time.perf_counter() - t0)
                logger.log_generation(row, self._terms_row(gen, ev))
                logger.log_stimulus_stats(gen, ev.stimulus_stats)
                every = cfg.metrics.snapshot_every
                if gen == last or (every > 0 and gen % every == 0):
                    logger.snapshot(gen, population[best].to_dict(), float(ev.fitness[best]))
                if progress:
                    progress(row)
                evaluated = population
                # Reproduce also after the last generation, so that the final checkpoint holds
                # generation `generations` and the run can be extended with --resume.
                population = self.reproduce(population, ev.fitness, gen)
                every = cfg.metrics.checkpoint_every
                if gen == last or (every > 0 and (gen + 1) % every == 0):
                    logger.checkpoint(gen + 1, [g.to_dict() for g in population], self.input_names)
        finally:
            self.close()
        return evaluated

    @staticmethod
    def _generation_row(gen, population, ev: Evaluation, best: int, elapsed: float) -> dict:
        f, s = ev.fitness, ev.stats
        return {
            "generation": gen,
            "fitness_mean": float(f.mean()),
            "fitness_max": float(f.max()),
            "fitness_min": float(f.min()),
            "fitness_var": float(f.var()),
            "n_species": "",  # speciation arrives with milestone 3
            "firing_rate_mean": float(s.firing_rate_hz.mean()),
            "energy_mean": float(s.final_energy.mean()),
            "energy_gained_mean": float(s.energy_gained.mean()),
            "energy_gained_max": float(s.energy_gained.max()),
            "alive_fraction": float((s.final_energy > 0).mean()),
            "visited_cells_mean": float(s.visited_cells.mean()),
            "hidden_mean": float(np.mean([g.n_hidden for g in population])),
            "synapses_mean": float(np.mean([g.n_synapses for g in population])),
            "best_hidden": population[best].n_hidden,
            "best_synapses": population[best].n_synapses,
            "elapsed_s": round(elapsed, 3),
        }

    @staticmethod
    def _terms_row(gen: int, ev: Evaluation) -> dict:
        row: dict = {"generation": gen}
        for t, v in ev.terms.items():
            row[f"{t}_mean"] = float(v.mean())
            row[f"{t}_min"] = float(v.min())
            row[f"{t}_max"] = float(v.max())
        return row


class Blinded:
    """Wraps a stimulus so that the agents perceive nothing from it."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def __setattr__(self, attr, value):
        if attr == "_inner":
            object.__setattr__(self, attr, value)
        else:
            setattr(self._inner, attr, value)

    def sense(self, agents, world):
        return np.zeros((len(agents), len(self._inner.channels)))


def load_population(path: str | Path) -> tuple[list[Genome], list[str]]:
    """Genomes and their input channel names from a run directory or a checkpoint file."""
    import json

    import yaml

    from ..config import config_from_dict

    path = Path(path)
    ckpt_path = path / "checkpoint.json" if path.is_dir() else path
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    names = data.get("input_names")
    if not names:  # older checkpoints: recompute from the run's config.yaml
        raw = yaml.safe_load((ckpt_path.parent / "config.yaml").read_text(encoding="utf-8"))
        raw.pop("_run", None)
        names = n_input_channels(config_from_dict(raw))[1]
    return [Genome.from_dict(d) for d in data["genomes"]], list(names)


def merge_evaluations(parts: list[Evaluation]) -> Evaluation:
    cat = np.concatenate
    return Evaluation(
        fitness=cat([p.fitness for p in parts]),
        terms={t: cat([p.terms[t] for p in parts]) for t in parts[0].terms},
        stimulus_stats={
            k: cat([p.stimulus_stats[k] for p in parts]) for k in parts[0].stimulus_stats
        },
        stats=BehaviourStats(
            **{
                f: cat([getattr(p.stats, f) for p in parts])
                for f in BehaviourStats.__dataclass_fields__
            }
        ),
    )


def limit_native_threads() -> None:
    """One process = one core: stop BLAS/OpenMP thread pools from oversubscribing CPUs."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


_WORKER: Evolution | None = None


def _init_worker(cfg: Config, blind: tuple[str, ...] = ()) -> None:
    global _WORKER
    _WORKER = Evolution(cfg, blind=blind)


def _evaluate_in_worker(genomes: list[Genome], generation: int, offset: int, total: int):
    assert _WORKER is not None
    return _WORKER.evaluate_chunk(genomes, generation, offset, total)
