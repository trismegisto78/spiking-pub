"""Load and dump configurations (YAML or TOML) into the dataclass schema."""

from __future__ import annotations

import dataclasses
import tomllib
import types
import typing
from pathlib import Path
from typing import Any

import yaml

from .schema import Config


class ConfigError(ValueError):
    pass


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path or 'config'}: unknown key(s) {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v, f"{path}.{k}" if path else k) for k, v in data.items()}
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, list | tuple) or len(value) != len(args):
            raise ConfigError(f"{path}: expected a list of {len(args)} values")
        return tuple(_coerce(a, v, path) for a, v in zip(args, value, strict=True))
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        (arg,) = typing.get_args(tp)
        return [_coerce(arg, v, path) for v in value]
    if origin is dict or origin is types.UnionType:
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = _build(Config, data, "")
    validate(cfg)
    phase_configs(cfg)  # validates the phases too
    return cfg


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def phase_configs(cfg: Config) -> list[tuple[int, str, Config]]:
    """[(first generation, name, config)] for every phase; phase 0 is the base config."""
    base = config_to_dict(cfg)
    base.pop("phases")
    out = [(0, "base", _build(Config, base, ""))]
    last = 0
    for i, ph in enumerate(cfg.phases):
        where = f"phases[{i}]"
        if set(ph) - {"from_generation", "name", "set"} or "from_generation" not in ph:
            raise ConfigError(f"{where}: keys are from_generation, name, set")
        start = ph["from_generation"]
        if not isinstance(start, int) or start <= last:
            raise ConfigError(f"{where}: from_generation must increase and be > 0")
        last = start
        changes = ph.get("set") or {}
        if not isinstance(changes, dict):
            raise ConfigError(f"{where}.set: expected a mapping")
        for forbidden in ("phases", "seed", "experiment", "genetics"):
            if forbidden in changes and not (
                forbidden == "genetics"
                and "generations" not in changes["genetics"]
                and "population" not in changes["genetics"]
            ):
                raise ConfigError(
                    f"{where}.set: cannot change {forbidden}"
                    + (" generations/population" if forbidden == "genetics" else "")
                )
        merged = _merge(base, changes)
        phase_cfg = _build(Config, merged, f"{where}.set")
        validate(phase_cfg)
        out.append((start, str(ph.get("name", f"phase{i + 1}")), phase_cfg))
    return out


def load_config(path: str | Path) -> Config:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".toml":
        data = tomllib.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    return config_from_dict(data)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    def convert(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(dataclasses.asdict(cfg))


def dump_config(cfg: Config, extra: dict[str, Any] | None = None) -> str:
    data = config_to_dict(cfg)
    if extra:
        data = {**data, **extra}
    return yaml.safe_dump(data, sort_keys=False)


def validate(cfg: Config) -> None:
    w, n, g = cfg.world, cfg.network, cfg.genetics
    checks = [
        (w.width > 0 and w.height > 0, "world size must be positive"),
        (w.steps_per_episode > 0, "world.steps_per_episode must be positive"),
        (w.arenas in ("isolated", "shared"), "world.arenas must be 'isolated' or 'shared'"),
        (n.encoder in ("rate", "latency", "current"), "network.encoder: rate|latency|current"),
        (n.dt_ms > 0, "network.dt_ms must be positive"),
        (n.steps_per_tick >= 1, "network.steps_per_tick must be >= 1"),
        (n.max_delay >= 1, "network.max_delay must be >= 1"),
        (0 <= n.hidden_init <= n.hidden_max, "need 0 <= network.hidden_init <= hidden_max"),
        (n.neuron.tau_m > 0, "network.neuron.tau_m must be positive"),
        (n.neuron.refractory_ms >= 0, "network.neuron.refractory_ms must be >= 0"),
        (g.population >= 2, "genetics.population must be >= 2"),
        (g.generations >= 1, "genetics.generations must be >= 1"),
        (1 <= g.tournament_size <= g.population, "genetics.tournament_size out of range"),
        (0 <= g.elitism < g.population, "genetics.elitism must be < population"),
        (0.0 <= g.crossover <= 1.0, "genetics.crossover must be a probability"),
        (g.encoding == "direct", "genetics.encoding: only 'direct' is implemented (milestone 1)"),
        (cfg.fitness.episodes >= 1, "fitness.episodes must be >= 1"),
        (n.action_outputs >= 0, "network.action_outputs must be >= 0"),
        (len(cfg.stimuli) >= 1, "at least one stimulus is required"),
        (all("type" in s for s in cfg.stimuli), "every stimulus needs a 'type'"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ConfigError(msg)
