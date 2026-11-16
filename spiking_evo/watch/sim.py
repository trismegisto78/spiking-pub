"""Live simulations for the GUI: step a world and describe it frame by frame.

Two kinds of session:

* ``GenerationalSession``: the genomes of a generational run (final population from
  ``checkpoint.json``, the best genome of a generation from ``snapshots/``, or a fresh random
  population) living episode after episode, exactly as during evaluation (same world, same
  controller), with episode keys ("watch", k) never used by evolution.
* ``EcosystemSession``: an ecosystem config started from its founders, step by step.

A session never writes to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from ..config import config_from_dict, load_config
from ..config.schema import Config
from ..core.clock import Clock
from ..core.world import World
from ..genetics.genome import Genome, build_network_batch
from ..genetics.population import Evolution, load_population
from ..seeding import derive_rng
from ..snn.controller import N_MOTOR_OUTPUTS, SNNController
from ..snn.encoders import make_encoder
from ..stimuli.food import FoodStimulus
from ..stimuli.hazard import HazardStimulus
from ..stimuli.registry import build_stimuli
from ..stimuli.season import SeasonStimulus
from ..stimuli.signal import SignalStimulus
from ..stimuli.terrain import TerrainStimulus

MOTOR_NAMES = ("avanti", "indietro", "gira_sx", "gira_dx")


def load_run_config(run: Path) -> Config:
    raw = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    raw.pop("_run", None)
    return config_from_dict(raw)


class Capture:
    """Wraps a controller and remembers the last observations and actions."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.obs: np.ndarray | None = None
        self.out: tuple = ()

    def __getattr__(self, name):  # feedback / energy_cost, when the inner controller has them
        return getattr(self.inner, name)

    def reset(self) -> None:
        self.inner.reset()

    def act(self, observations, active):
        self.obs = np.array(observations, copy=True)
        self.out = self.inner.act(observations, active)
        return self.out


class Session:
    """Common part: frames and agent details from ``self.world``, ``self.capture``,
    ``self.snn`` (the SNNController) and ``self.genomes`` (one per agent)."""

    kind = "?"
    world: World
    capture: Capture
    input_names: list[str]
    genomes: list[Genome]
    label: str

    @property
    def snn(self) -> SNNController:
        raise NotImplementedError

    # -- description ------------------------------------------------------------------
    def info(self) -> dict:
        w = self.world
        return {
            "kind": self.kind,
            "label": self.label,
            "width": w.width,
            "height": w.height,
            "arenas": w.n_arenas,
            "agents": w.n_agents,
            "inputs": self.input_names,
            "actions": w.n_actions,
            "stimuli": [type(s).__name__ for s in w.stimuli],
        }

    def layers(self, arena: int) -> dict:
        w, t = self.world, self.world.clock.tick
        out: dict[str, dict] = {}
        for s in w.stimuli:
            if isinstance(s, SignalStimulus):
                out[s.name] = {"kind": "signal", "data": s._grid[arena].ravel().tolist()}
            elif isinstance(s, FoodStimulus):
                out[s.name] = {"kind": "food", "data": s.grid[arena].ravel().tolist()}
            elif isinstance(s, HazardStimulus):
                g = np.asarray(s.grid[arena], dtype=float)
                out[s.name] = {"kind": "hazard", "data": np.round(g, 2).ravel().tolist()}
            elif isinstance(s, TerrainStimulus):
                cells = {
                    "soil": np.asarray(s.variety[arena]),
                    "crop": (np.asarray(s.sown[arena]) > 0).astype(int),
                    "poison": (np.asarray(s.poison_until[arena]) > t).astype(int),
                }
                for kind, grid in cells.items():
                    out[f"{s.name}.{kind}"] = {"kind": kind, "data": grid.ravel().tolist()}
        return out

    def season(self) -> str | None:
        for s in self.world.stimuli:
            if isinstance(s, SeasonStimulus):
                return s.season_names[s.current]
        return None

    def emitting(self) -> np.ndarray:
        a = self.world.agents
        for s in self.world.stimuli:
            if isinstance(s, SignalStimulus) and a.action.shape[1] > s.action:
                return a.action[:, s.action] & a.alive
        return np.zeros(len(a), dtype=bool)

    def agent_rows(self, idx: np.ndarray) -> list[dict]:
        a = self.world.agents
        emit = self.emitting()
        rows = []
        for i in idx:
            rows.append(
                {
                    "i": int(i),
                    "x": int(a.x[i]),
                    "y": int(a.y[i]),
                    "h": int(a.heading[i]),
                    "e": round(float(a.energy[i]), 2),
                    "alive": bool(a.alive[i]),
                    "stage": int(a.stage[i]),
                    "emit": bool(emit[i]),
                    "act": [bool(v) for v in a.action[i]] if a.action.size else [],
                    **self.extra_row(int(i)),
                }
            )
        return rows

    def extra_row(self, i: int) -> dict:
        return {}

    def visible(self, arena: int) -> np.ndarray:
        a = self.world.agents
        return np.flatnonzero(a.arena == arena)

    def frame(self, arena: int) -> dict:
        arena = int(np.clip(arena, 0, self.world.n_arenas - 1))
        a = self.world.agents
        idx = self.visible(arena)
        return {
            "tick": int(self.world.clock.tick),
            "arena": arena,
            "season": self.season(),
            "alive": int(a.alive[idx].sum()),
            "energy_gained": round(float(a.energy_gained[idx].mean()) if len(idx) else 0.0, 2),
            "agents": self.agent_rows(idx),
            "layers": self.layers(arena),
            **self.extra_frame(),
        }

    def extra_frame(self) -> dict:
        return {}

    def agent(self, i: int) -> dict:
        a = self.world.agents
        g = self.genomes[i]
        n = g.n_neurons
        spikes = self.snn._tick_spikes[i, :n].tolist() if i < len(self.snn._tick_spikes) else []
        obs = self.capture.obs[i].tolist() if self.capture.obs is not None else []
        if obs and len(obs) == len(self.input_names) - 1 and self.input_names[-1] == "bias":
            obs.append(1.0)  # the controller adds the constant bias input itself
        motor = {}
        if self.capture.out:
            fwd, turn = self.capture.out[0], self.capture.out[1]
            motor = {"forward": int(fwd[i]), "turn": int(turn[i])}
        return {
            "i": i,
            "x": int(a.x[i]),
            "y": int(a.y[i]),
            "heading": int(a.heading[i]),
            "energy": float(a.energy[i]),
            "energy_gained": float(a.energy_gained[i]),
            "alive": bool(a.alive[i]),
            "age": int(a.age[i]),
            "stage": int(a.stage[i]),
            "arena": int(a.arena[i]),
            "inputs": dict(zip(self.input_names, obs, strict=False)),
            "motor": motor,
            "actions": [bool(v) for v in a.action[i]] if a.action.size else [],
            "brain": {
                "n_inputs": g.n_inputs,
                "n_outputs": g.n_outputs,
                "n_neurons": n,
                "n_hidden": g.n_hidden,
                "input_names": self.input_names,
                "output_names": [
                    *MOTOR_NAMES,
                    *(f"azione{k}" for k in range(g.n_outputs - N_MOTOR_OUTPUTS)),
                ],
                "src": g.src.tolist(),
                "dst": g.dst.tolist(),
                "weight": np.round(g.weight, 3).tolist(),
                "thresh": np.round(g.v_thresh, 3).tolist(),
                "dormant": g.dormant.astype(int).tolist(),
                "spikes": spikes,
            },
            **self.extra_agent(i),
        }

    def extra_agent(self, i: int) -> dict:
        return {}


class GenerationalSession(Session):
    kind = "generations"

    def __init__(self, cfg: Config, genomes: list[Genome], label: str) -> None:
        self.cfg = cfg
        self.label = label
        evo = Evolution(cfg)
        self.input_names = evo.input_names
        genomes = [g.repair(evo.spec) for g in genomes]
        k = cfg.genetics.clones
        self.genomes = [g for g in genomes for _ in range(k)] if k > 1 else genomes
        self.net = build_network_batch(self.genomes, evo.spec)
        self.world = World(
            cfg.world,
            cfg.agent,
            build_stimuli(cfg.stimuli),
            n_agents=len(self.genomes),
            clock=Clock(cfg.network.dt_ms, cfg.network.steps_per_tick),
            n_actions=cfg.network.action_outputs,
        )
        self.encoder = make_encoder(
            cfg.network.encoder,
            max_rate_hz=cfg.network.max_rate_hz,
            dt_ms=cfg.network.dt_ms,
            current_gain=cfg.network.current_gain,
        )
        self.episode = -1
        self.new_episode()

    @property
    def snn(self) -> SNNController:
        return self.controller

    def new_episode(self) -> None:
        cfg = self.cfg
        self.episode += 1
        keys = ("watch", self.episode)
        self.controller = SNNController(
            self.net,
            self.encoder,
            dt_ms=cfg.network.dt_ms,
            steps_per_tick=cfg.network.steps_per_tick,
            bias_input=cfg.network.bias_input,
            motor_threshold_hz=cfg.network.motor_threshold_hz,
            rng=derive_rng(cfg.seed, *keys, "encoder"),
            action_threshold_hz=cfg.network.action_threshold_hz,
            plasticity=cfg.network.plasticity if cfg.network.plasticity.enabled else None,
            development=cfg.network.development if cfg.network.development.enabled else None,
            weight_max=cfg.network.weight_max,
        )
        self.capture = Capture(self.controller)
        self.world.reset(cfg.seed, *keys)
        self.capture.reset()

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            done = self.world.clock.tick >= self.cfg.world.steps_per_episode
            if done or not self.world.agents.alive.any():
                self.new_episode()
            self.world.step(self.capture)

    def extra_frame(self) -> dict:
        return {
            "episode": self.episode,
            "steps_per_episode": self.cfg.world.steps_per_episode,
        }

    def extra_row(self, i: int) -> dict:
        k = self.cfg.genetics.clones
        return {"genome": i // k if k > 1 else i}


class EcosystemSession(Session):
    kind = "ecosystem"

    def __init__(self, cfg: Config, label: str) -> None:
        from ..ecosystem import Ecosystem

        self.cfg = cfg
        self.label = label
        self.sim = Ecosystem(cfg)
        self.capture = Capture(self.sim.wrapper)
        self.sim.wrapper = self.capture
        self.world = self.sim.world
        self.input_names = self.sim.input_names

    @property
    def genomes(self) -> list[Genome]:
        return self.sim.genomes

    @property
    def snn(self) -> SNNController:
        return self.sim.controller

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.world.agents.alive.any():
                self.sim.step()

    def visible(self, arena: int) -> np.ndarray:
        return np.flatnonzero(self.world.agents.alive)

    def extra_row(self, i: int) -> dict:
        s = self.sim
        return {"id": int(s.ids[i]), "parent": int(s.parent[i])}

    def extra_frame(self) -> dict:
        s = self.sim
        alive = s.world.agents.alive
        return {
            "population": int(alive.sum()),
            "juveniles": int((s.world.agents.stage == 1).sum()),
            "max_generation": int(s.generation[alive].max()) if alive.any() else 0,
        }

    def extra_agent(self, i: int) -> dict:
        s = self.sim
        return {
            "id": int(s.ids[i]),
            "parent": int(s.parent[i]),
            "lineage": int(s.lineage[i]),
            "generation": int(s.generation[i]),
            "traits": {k: float(v) for k, v in s.traits.row(i).items()},
        }


def open_session(source: dict) -> Session:
    """``source``: {"run": path, "genomes": "checkpoint" | "random" | <snapshot file>} for a
    generational run, {"run": path} for an ecosystem run (started from its config), or
    {"config": path} for any experiment file (fresh random population or ecosystem)."""
    if "config" in source:
        cfg = load_config(source["config"])
        label = Path(source["config"]).stem
        if cfg.ecosystem.ticks and "ecosystem" in _raw_keys(Path(source["config"])):
            return EcosystemSession(cfg, label)
        return GenerationalSession(cfg, Evolution(cfg).initial_population(), f"{label} (casuale)")
    run = Path(source["run"])
    cfg = load_run_config(run)
    label = f"{run.parent.name}/{run.name}"
    if (run / "ecosystem.csv").exists():
        return EcosystemSession(cfg, label)
    what = source.get("genomes", "checkpoint")
    if what == "random":
        return GenerationalSession(cfg, Evolution(cfg).initial_population(), f"{label} casuale")
    if what == "checkpoint":
        genomes, _ = load_population(run)
        return GenerationalSession(cfg, genomes, f"{label} popolazione finale")
    snap = run / "snapshots" / Path(what).name
    data = json.loads(snap.read_text(encoding="utf-8"))
    best = Genome.from_dict(data.get("genome", data))
    copies = cfg.world.group_size if cfg.world.arenas == "groups" else 1
    if cfg.genetics.clones > 1:
        copies = 1  # the session itself makes the clones
    gen = data.get("generation", "?")
    return GenerationalSession(cfg, [best] * copies, f"{label} migliore gen {gen}")


def _raw_keys(path: Path) -> set[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return set(raw)
