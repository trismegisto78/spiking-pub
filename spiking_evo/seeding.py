"""Deterministic derivation of sub-seeds from the global seed.

Every stochastic component asks for its own generator with a stable key, e.g.
``derive_rng(seed, "eval", generation, "stimulus", 0)``. Generators derived with
different keys are statistically independent, and a given key always yields the same
stream, so any generation can be reproduced in isolation.
"""

from __future__ import annotations

import zlib

import numpy as np


def _key_to_int(key: int | str) -> int:
    if isinstance(key, bool):
        raise TypeError("bool is not a valid seed key")
    if isinstance(key, int):
        if key < 0:
            raise ValueError("seed keys must be non-negative")
        return key
    # crc32 is stable across processes and Python versions (unlike hash()).
    return zlib.crc32(key.encode("utf-8"))


def derive_seed_sequence(seed: int, *keys: int | str) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy=seed, spawn_key=tuple(_key_to_int(k) for k in keys))


def derive_rng(seed: int, *keys: int | str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed_sequence(seed, *keys)))
