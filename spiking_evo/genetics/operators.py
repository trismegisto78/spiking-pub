"""Genetic operators: tournament selection, crossover and mutation.

All operators are pure (they return new genomes) and end with ``Genome.repair``, so they
can never produce an invalid network.
"""

from __future__ import annotations

import numpy as np

from ..config.schema import MutationConfig
from .genome import Genome, GenomeSpec


def tournament_select(
    fitness: np.ndarray, tournament_size: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of ``n`` winners, each the fittest of ``tournament_size`` random entrants.

    Entrants of one tournament are distinct; ties go to the lowest index.
    """
    pop = len(fitness)
    entrants = np.argsort(rng.random((n, pop)), axis=1)[:, :tournament_size]
    best = np.argmax(fitness[entrants], axis=1)
    return entrants[np.arange(n), best]


def crossover(fitter: Genome, other: Genome, spec: GenomeSpec, rng: np.random.Generator) -> Genome:
    """Uniform crossover aligned by neuron index and by (src, dst) synapse key.

    The child has the neuron count of the fitter parent. Neuron parameters and matching
    synapses are taken from either parent with probability 1/2; synapses present only in
    the fitter parent are inherited, those present only in the other parent are not.
    """
    child = fitter.copy()
    m = min(fitter.n_neurons, other.n_neurons)
    take = rng.random(m) < 0.5
    for name in ("tau_m", "v_thresh", "v_reset", "refractory_ms"):
        getattr(child, name)[:m][take] = getattr(other, name)[:m][take]
    big = spec.n_max
    key_a = fitter.src * big + fitter.dst
    key_b = other.src * big + other.dst
    common, ia, ib = np.intersect1d(key_a, key_b, assume_unique=True, return_indices=True)
    swap = rng.random(common.size) < 0.5
    child.weight[ia[swap]] = other.weight[ib[swap]]
    child.delay[ia[swap]] = other.delay[ib[swap]]
    return child.repair(spec)


def mutate(genome: Genome, spec: GenomeSpec, m: MutationConfig, rng: np.random.Generator) -> Genome:
    g = genome.copy()
    nf = spec.n_inputs  # input neurons have no evolvable parameters

    # Per-gene parameter mutations.
    hit = rng.random(g.n_synapses) < m.weight
    g.weight[hit] += rng.normal(0.0, m.weight_sigma, size=int(hit.sum()))
    hit = rng.random(g.n_synapses) < m.delay
    g.delay[hit] += rng.choice(np.array([-1, 1]), size=int(hit.sum()))
    hit = rng.random(g.n_neurons - nf) < m.threshold
    g.v_thresh[nf:][hit] += rng.normal(0.0, m.threshold_sigma, size=int(hit.sum()))
    hit = rng.random(g.n_neurons - nf) < m.tau
    g.tau_m[nf:][hit] *= np.exp(rng.normal(0.0, m.tau_sigma, size=int(hit.sum())))
    g = g.repair(spec)

    # Per-genome structural mutations.
    if rng.random() < m.add_synapse:
        g = add_synapse(g, spec, rng)
    if rng.random() < m.remove_synapse:
        g = remove_synapse(g, spec, rng)
    if rng.random() < m.add_neuron:
        g = add_neuron(g, spec, rng)
    if rng.random() < m.remove_neuron:
        g = remove_neuron(g, spec, rng)
    return g


def add_synapse(g: Genome, spec: GenomeSpec, rng: np.random.Generator, tries: int = 20) -> Genome:
    n = g.n_neurons
    existing = set(zip(g.src.tolist(), g.dst.tolist(), strict=True))
    for _ in range(tries):
        s = int(rng.integers(n))
        d = int(rng.integers(g.n_inputs, n))
        if (s, d) not in existing:
            return _with_synapses(
                g,
                spec,
                [s],
                [d],
                [rng.normal(0.0, spec.init_weight_sigma)],
                [rng.integers(1, spec.max_delay + 1)],
            )
    return g


def remove_synapse(g: Genome, spec: GenomeSpec, rng: np.random.Generator) -> Genome:
    if g.n_synapses == 0:
        return g
    keep = np.ones(g.n_synapses, dtype=bool)
    keep[rng.integers(g.n_synapses)] = False
    return _select_synapses(g, keep).repair(spec)


def add_neuron(g: Genome, spec: GenomeSpec, rng: np.random.Generator) -> Genome:
    """Add a hidden neuron by splitting a random synapse src -> dst into
    src -> new -> dst (NEAT style). Without synapses, the neuron gets one random input
    and one random output connection."""
    if g.n_hidden >= spec.hidden_max:
        return g
    new = g.n_neurons
    h = g.copy()
    for name, val in (
        ("tau_m", spec.tau_m),
        ("v_thresh", spec.v_thresh),
        ("v_reset", spec.v_reset),
        ("refractory_ms", spec.refractory_ms),
    ):
        setattr(h, name, np.append(getattr(h, name), val))
    if h.n_synapses > 0:
        i = int(rng.integers(h.n_synapses))
        s, d, w, dl = int(h.src[i]), int(h.dst[i]), float(h.weight[i]), int(h.delay[i])
        keep = np.ones(h.n_synapses, dtype=bool)
        keep[i] = False
        h = _select_synapses(h, keep)
        w_in = 1.5 * spec.v_thresh  # reliably relays the original spikes
    else:
        s = int(rng.integers(new))
        d = int(rng.integers(h.n_inputs, new + 1))
        w = rng.normal(0.0, spec.init_weight_sigma)
        dl = int(rng.integers(1, spec.max_delay + 1))
        w_in = rng.normal(0.0, spec.init_weight_sigma)
    return _with_synapses(h, spec, [s, new], [new, d], [w_in, w], [1, dl])


def remove_neuron(g: Genome, spec: GenomeSpec, rng: np.random.Generator) -> Genome:
    if g.n_hidden == 0:
        return g
    k = int(rng.integers(spec.n_fixed, g.n_neurons))
    h = g.copy()
    for name in ("tau_m", "v_thresh", "v_reset", "refractory_ms"):
        setattr(h, name, np.delete(getattr(h, name), k))
    keep = (h.src != k) & (h.dst != k)
    h = _select_synapses(h, keep)
    h.src = np.where(h.src > k, h.src - 1, h.src)
    h.dst = np.where(h.dst > k, h.dst - 1, h.dst)
    return h.repair(spec)


def _select_synapses(g: Genome, keep: np.ndarray) -> Genome:
    h = g.copy()
    h.src, h.dst, h.weight, h.delay = h.src[keep], h.dst[keep], h.weight[keep], h.delay[keep]
    return h


def _with_synapses(g: Genome, spec: GenomeSpec, src, dst, weight, delay) -> Genome:
    h = g.copy()
    h.src = np.concatenate([h.src, np.asarray(src, dtype=np.int64)])
    h.dst = np.concatenate([h.dst, np.asarray(dst, dtype=np.int64)])
    h.weight = np.concatenate([h.weight, np.asarray(weight, dtype=np.float64)])
    h.delay = np.concatenate([h.delay, np.asarray(delay, dtype=np.int64)])
    return h.repair(spec)
