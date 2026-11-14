"""Stimulus registry.

Adding a stimulus = create a module in ``spiking_evo/stimuli/`` with a class decorated by
``@register("<type>")``. Modules of this package are imported automatically, so neither
this file nor ``core/`` needs to change. Config entries look like
``{"type": "<type>", **constructor_kwargs}``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from .base import Stimulus

_REGISTRY: dict[str, type] = {}
_DISCOVERED = False


def register(type_name: str):
    def deco(cls: type) -> type:
        if type_name in _REGISTRY and _REGISTRY[type_name] is not cls:
            raise ValueError(f"stimulus type {type_name!r} already registered")
        _REGISTRY[type_name] = cls
        return cls

    return deco


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module(__package__)
    for mod in pkgutil.iter_modules(package.__path__):
        if mod.name not in ("base", "registry"):
            importlib.import_module(f"{__package__}.{mod.name}")
    _DISCOVERED = True


def available() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def build_stimulus(spec: dict[str, Any]) -> Stimulus:
    _discover()
    spec = dict(spec)
    type_name = spec.pop("type")
    if type_name not in _REGISTRY:
        raise ValueError(f"unknown stimulus type {type_name!r}; available: {available()}")
    cls = _REGISTRY[type_name]
    params = inspect.signature(cls).parameters
    unknown = set(spec) - set(params)
    if unknown:
        raise ValueError(f"stimulus {type_name!r}: unknown parameter(s) {sorted(unknown)}")
    stim = cls(**spec)
    if not isinstance(stim, Stimulus):
        raise TypeError(f"{cls.__name__} does not implement the Stimulus protocol")
    return stim


def build_stimuli(specs: list[dict[str, Any]]) -> list[Stimulus]:
    stimuli = [build_stimulus(s) for s in specs]
    names = [s.name for s in stimuli]
    if len(set(names)) != len(names):
        raise ValueError(f"stimulus names must be unique, got {names}; set 'name' explicitly")
    return stimuli
