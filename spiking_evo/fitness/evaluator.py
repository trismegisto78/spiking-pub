"""Fitness as a weighted sum of separately logged terms.

Terms (already signed: penalties are <= 0):

    energy           weight * energy gained from stimuli during the agent's life
    firing_penalty  -weight * max(0, rate - target) / target     (saturated networks)
    silence_penalty -weight * [no non-input neuron ever fired]    (dead networks)
    exploration      weight * number of distinct cells visited

A weight of 0 disables a term (it is still logged, as 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import FitnessConfig

TERM_NAMES = ("energy", "firing_penalty", "silence_penalty", "exploration")


@dataclass
class BehaviourStats:
    energy_gained: np.ndarray
    final_energy: np.ndarray
    firing_rate_hz: np.ndarray
    noninput_spikes: np.ndarray
    visited_cells: np.ndarray
    steps_alive: np.ndarray


def compute_terms(cfg: FitnessConfig, s: BehaviourStats) -> dict[str, np.ndarray]:
    fp = cfg.firing_penalty
    target = max(fp.target_rate_hz, 1e-9)
    return {
        "energy": cfg.energy * s.energy_gained,
        "firing_penalty": -fp.weight * np.maximum(0.0, s.firing_rate_hz - target) / target,
        "silence_penalty": -cfg.silence_penalty.weight * (s.noninput_spikes == 0),
        "exploration": cfg.exploration.weight * s.visited_cells,
    }


def total_fitness(terms: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum([terms[name] for name in TERM_NAMES], axis=0).astype(np.float64)
