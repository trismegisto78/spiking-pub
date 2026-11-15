"""Direct genetic encoding of a spiking network.

A genome lists its neurons (per-neuron LIF parameters, laid out as inputs, outputs, then
hidden neurons) and its synapses (source, destination, weight, delay). ``repair`` brings any
genome back to a valid network: weights and parameters are clipped, delays rounded and
clipped, synapses towards missing neurons or into input neurons are dropped, duplicated
(src, dst) pairs are merged. Every genetic operator ends with ``repair``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config.schema import Config
from ..snn.network import NetworkBatch

GENOME_FORMAT = "spiking_evo.genome"
GENOME_VERSION = 3  # v2: learning_rate, plastic flags; v3: dormant, wake_at. Older load.


@dataclass(frozen=True)
class GenomeSpec:
    """Constants shared by all genomes of an experiment."""

    n_inputs: int
    n_outputs: int
    hidden_max: int
    max_delay: int
    weight_max: float
    thresh_range: tuple[float, float]
    tau_range: tuple[float, float]
    tau_m: float
    v_thresh: float
    v_reset: float
    refractory_ms: float
    init_hidden: int
    init_connection_prob: float
    init_weight_sigma: float
    plastic: bool = False  # plasticity enabled in this experiment
    init_learning_rate: float = 0.0
    learning_rate_max: float = 1.0
    init_plastic_prob: float = 0.0
    development: bool = False
    init_dormant_prob: float = 0.0
    init_wake_at: float = 0.0

    @classmethod
    def from_config(cls, cfg: Config, n_inputs: int, n_outputs: int) -> GenomeSpec:
        n = cfg.network
        return cls(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_max=n.hidden_max,
            max_delay=n.max_delay,
            weight_max=n.weight_max,
            thresh_range=tuple(n.thresh_range),
            tau_range=tuple(n.tau_range),
            tau_m=n.neuron.tau_m,
            v_thresh=n.neuron.v_thresh,
            v_reset=n.neuron.v_reset,
            refractory_ms=n.neuron.refractory_ms,
            init_hidden=n.hidden_init,
            init_connection_prob=n.init_connection_prob,
            init_weight_sigma=n.init_weight_sigma,
            plastic=n.plasticity.enabled,
            init_learning_rate=n.plasticity.init_learning_rate,
            learning_rate_max=n.plasticity.learning_rate_max,
            init_plastic_prob=n.plasticity.init_plastic_prob,
            development=n.development.enabled,
            init_dormant_prob=n.development.init_dormant_prob,
            init_wake_at=n.development.init_wake_at,
        )

    @property
    def n_fixed(self) -> int:
        return self.n_inputs + self.n_outputs

    @property
    def n_max(self) -> int:
        return self.n_fixed + self.hidden_max


@dataclass
class Genome:
    n_inputs: int
    n_outputs: int
    tau_m: np.ndarray  # (N,)
    v_thresh: np.ndarray  # (N,)
    v_reset: np.ndarray  # (N,)
    refractory_ms: np.ndarray  # (N,)
    src: np.ndarray  # (S,) int
    dst: np.ndarray  # (S,) int
    weight: np.ndarray  # (S,) float
    delay: np.ndarray  # (S,) int
    plastic: np.ndarray | None = None  # (S,) bool: may this synapse change during life
    learning_rate: float = 0.0
    dormant: np.ndarray | None = None  # (N,) bool: hidden neuron asleep at birth
    wake_at: np.ndarray | None = None  # (N,) float: difficulty that wakes it up

    def __post_init__(self) -> None:
        if self.plastic is None:
            self.plastic = np.zeros(len(self.src), dtype=bool)
        if self.dormant is None:
            self.dormant = np.zeros(len(self.tau_m), dtype=bool)
        if self.wake_at is None:
            self.wake_at = np.zeros(len(self.tau_m))

    # -- sizes ------------------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return len(self.tau_m)

    @property
    def n_hidden(self) -> int:
        return self.n_neurons - self.n_inputs - self.n_outputs

    @property
    def n_synapses(self) -> int:
        return len(self.src)

    # -- construction -------------------------------------------------------------
    @classmethod
    def random(cls, spec: GenomeSpec, rng: np.random.Generator) -> Genome:
        n = spec.n_fixed + spec.init_hidden
        pairs_src, pairs_dst = np.meshgrid(np.arange(n), np.arange(spec.n_inputs, n), indexing="ij")
        keep = rng.random(pairs_src.shape) < spec.init_connection_prob
        src, dst = pairs_src[keep], pairs_dst[keep]
        g = cls(
            n_inputs=spec.n_inputs,
            n_outputs=spec.n_outputs,
            tau_m=np.full(n, spec.tau_m),
            v_thresh=np.full(n, spec.v_thresh),
            v_reset=np.full(n, spec.v_reset),
            refractory_ms=np.full(n, spec.refractory_ms),
            src=src,
            dst=dst,
            weight=rng.normal(0.0, spec.init_weight_sigma, size=src.size),
            delay=rng.integers(1, spec.max_delay + 1, size=src.size),
        )
        if spec.development:  # extra draws only when development is on
            hidden = np.arange(n) >= spec.n_fixed
            g.dormant = hidden & (rng.random(n) < spec.init_dormant_prob)
            g.wake_at = np.full(n, spec.init_wake_at)
        if spec.plastic:  # extra draws only when plasticity is on (keeps old runs identical)
            g.plastic = rng.random(src.size) < spec.init_plastic_prob
            g.learning_rate = spec.init_learning_rate
        return g.repair(spec)

    def copy(self) -> Genome:
        return Genome(
            self.n_inputs,
            self.n_outputs,
            *(a.copy() for a in self._arrays()),
            learning_rate=self.learning_rate,
            dormant=self.dormant.copy(),
            wake_at=self.wake_at.copy(),
        )

    def _arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.tau_m,
            self.v_thresh,
            self.v_reset,
            self.refractory_ms,
            self.src,
            self.dst,
            self.weight,
            self.delay,
            self.plastic,
        )

    def repair(self, spec: GenomeSpec) -> Genome:
        """Return a valid copy of this genome (see module docstring)."""
        if self.n_inputs != spec.n_inputs or self.n_outputs != spec.n_outputs:
            raise ValueError("genome does not match the experiment's input/output layout")
        n = min(self.n_neurons, spec.n_max)
        if n < spec.n_fixed:
            raise ValueError("genome is missing input/output neurons")
        tau = np.clip(np.asarray(self.tau_m[:n], float), *spec.tau_range)
        thresh = np.clip(np.asarray(self.v_thresh[:n], float), *spec.thresh_range)
        reset = np.minimum(np.asarray(self.v_reset[:n], float), thresh - 1e-6)
        refr = np.maximum(np.asarray(self.refractory_ms[:n], float), 0.0)
        dormant = np.zeros(n, dtype=bool)
        wake_at = np.zeros(n)
        if spec.development:
            dm = np.asarray(self.dormant, dtype=bool)
            wa = np.asarray(self.wake_at, dtype=float)
            if dm.shape == (self.n_neurons,) and wa.shape == (self.n_neurons,):
                dormant = dm[:n].copy()
                wake_at = np.maximum(wa[:n], 0.0)
            dormant[: spec.n_fixed] = False  # inputs and outputs are always awake

        src = np.asarray(self.src, dtype=np.int64)
        dst = np.asarray(self.dst, dtype=np.int64)
        w = np.asarray(self.weight, dtype=np.float64)
        d = np.rint(np.asarray(self.delay, dtype=np.float64))
        pl = np.asarray(self.plastic, dtype=bool)
        if pl.shape != src.shape:  # synapse arrays replaced by hand: no plastic flags
            pl = np.zeros(src.shape, dtype=bool)
        ok = (src >= 0) & (src < n) & (dst >= self.n_inputs) & (dst < n) & np.isfinite(w)
        src, dst, w, d, pl = src[ok], dst[ok], w[ok], d[ok], pl[ok]
        # Canonical order and at most one synapse per (src, dst) pair.
        key = src * n + dst
        _, first = np.unique(key, return_index=True)
        first = np.sort(first)
        order = np.lexsort((dst[first], src[first]))
        idx = first[order]
        return Genome(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            tau_m=tau,
            v_thresh=thresh,
            v_reset=reset,
            refractory_ms=refr,
            src=src[idx],
            dst=dst[idx],
            weight=np.clip(w[idx], -spec.weight_max, spec.weight_max),
            delay=np.clip(d[idx], 1, spec.max_delay).astype(np.int64),
            plastic=pl[idx] & spec.plastic,
            learning_rate=float(np.clip(self.learning_rate, 0.0, spec.learning_rate_max))
            if spec.plastic
            else 0.0,
            dormant=dormant,
            wake_at=wake_at,
        )

    # -- comparison / serialisation ---------------------------------------------------
    def equals(self, other: Genome) -> bool:
        return (
            self.n_inputs == other.n_inputs
            and self.n_outputs == other.n_outputs
            and all(
                a.shape == b.shape and np.array_equal(a, b)
                for a, b in zip(self._arrays(), other._arrays(), strict=True)
            )
            and self.learning_rate == other.learning_rate
            and np.array_equal(self.dormant, other.dormant)
            and np.array_equal(self.wake_at, other.wake_at)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": GENOME_FORMAT,
            "version": GENOME_VERSION,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "neurons": {
                "tau_m": self.tau_m.tolist(),
                "v_thresh": self.v_thresh.tolist(),
                "v_reset": self.v_reset.tolist(),
                "refractory_ms": self.refractory_ms.tolist(),
                "dormant": self.dormant.astype(int).tolist(),
                "wake_at": self.wake_at.tolist(),
            },
            "synapses": {
                "src": self.src.tolist(),
                "dst": self.dst.tolist(),
                "weight": self.weight.tolist(),
                "delay": self.delay.tolist(),
                "plastic": self.plastic.astype(int).tolist(),
            },
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genome:
        if data.get("format") != GENOME_FORMAT:
            raise ValueError("not a spiking_evo genome")
        if data.get("version") not in (1, 2, GENOME_VERSION):
            raise ValueError(f"unsupported genome version {data.get('version')}")
        nr, sy = data["neurons"], data["synapses"]
        return cls(
            n_inputs=int(data["n_inputs"]),
            n_outputs=int(data["n_outputs"]),
            tau_m=np.array(nr["tau_m"], dtype=np.float64),
            v_thresh=np.array(nr["v_thresh"], dtype=np.float64),
            v_reset=np.array(nr["v_reset"], dtype=np.float64),
            refractory_ms=np.array(nr["refractory_ms"], dtype=np.float64),
            src=np.array(sy["src"], dtype=np.int64),
            dst=np.array(sy["dst"], dtype=np.int64),
            weight=np.array(sy["weight"], dtype=np.float64),
            delay=np.array(sy["delay"], dtype=np.int64),
            plastic=np.array(sy.get("plastic", [0] * len(sy["src"])), dtype=bool),
            learning_rate=float(data.get("learning_rate", 0.0)),
            dormant=np.array(nr["dormant"], dtype=bool) if "dormant" in nr else None,
            wake_at=np.array(nr["wake_at"], dtype=np.float64) if "wake_at" in nr else None,
        )


def build_network_batch(genomes: list[Genome], spec: GenomeSpec) -> NetworkBatch:
    """Pad a list of genomes into one batched network (one network per genome)."""
    net = NetworkBatch.empty(
        batch=len(genomes),
        n_inputs=spec.n_inputs,
        n_outputs=spec.n_outputs,
        n_hidden=spec.hidden_max,
        max_delay=spec.max_delay,
        tau_m=spec.tau_m,
        v_thresh=spec.v_thresh,
        v_reset=spec.v_reset,
        refractory_ms=spec.refractory_ms,
    )
    net.active[:] = False
    if spec.development:
        net.dormant = np.zeros(net.active.shape, dtype=bool)
        net.wake_at = np.zeros(net.active.shape)
    for b, g in enumerate(genomes):
        n = g.n_neurons
        net.tau_m[b, :n] = g.tau_m
        net.v_thresh[b, :n] = g.v_thresh
        net.v_reset[b, :n] = g.v_reset
        net.refractory_ms[b, :n] = g.refractory_ms
        net.active[b, :n] = True
        if spec.development:
            net.dormant[b, :n] = g.dormant
            net.wake_at[b, :n] = g.wake_at
        net.weights[b, g.delay - 1, g.src, g.dst] = g.weight
    if spec.plastic:
        net.plastic = np.zeros(net.weights.shape, dtype=bool)
        net.learning_rate = np.array([g.learning_rate for g in genomes])
        for b, g in enumerate(genomes):
            net.plastic[b, g.delay - 1, g.src, g.dst] = g.plastic
    return net


def adapt_genome(
    genome: Genome,
    old_inputs: list[str],
    new_inputs: list[str],
    new_outputs: int,
    spec: GenomeSpec,
    rng: np.random.Generator | None = None,
) -> Genome:
    """Carry a genome over to a new sensory/motor layout (environment change).

    Inputs are matched by channel name: a sense that still exists keeps its synapses, a new
    sense gets a fresh input neuron with no synapses, a sense that disappeared loses its
    synapses. Outputs are matched by position (the first 4 are always movement); new outputs
    start unconnected, removed ones lose their synapses. Hidden neurons and all their
    parameters are kept.

    With ``rng``, every new input and output neuron is wired at random like in a newborn
    genome (each possible synapse with probability ``spec.init_connection_prob``), so that a
    new sense or muscle is used from the start and evolution can refine it. Without
    wiring a new output never fires and a new sense is ignored.
    """
    if len(old_inputs) != genome.n_inputs:
        raise ValueError("old_inputs does not match the genome's input count")
    n_in_old, n_out_old = genome.n_inputs, genome.n_outputs
    n_in_new = len(new_inputs)
    pos = {name: i for i, name in enumerate(new_inputs)}
    n_hidden = genome.n_hidden
    n_new = n_in_new + new_outputs + n_hidden

    # Old index -> new index (-1 = gone).
    mapping = np.full(genome.n_neurons, -1, dtype=np.int64)
    for i, name in enumerate(old_inputs):
        mapping[i] = pos.get(name, -1)
    for o in range(min(n_out_old, new_outputs)):
        mapping[n_in_old + o] = n_in_new + o
    mapping[n_in_old + n_out_old :] = n_in_new + new_outputs + np.arange(n_hidden)

    def params(old: np.ndarray, default: float) -> np.ndarray:
        out = np.full(n_new, default)
        keep = mapping >= 0
        out[mapping[keep]] = old[keep]
        return out

    src, dst = mapping[genome.src], mapping[genome.dst]
    ok = (src >= 0) & (dst >= 0)
    src, dst = src[ok], dst[ok]
    weight, delay = genome.weight[ok], genome.delay[ok]
    pl = np.asarray(genome.plastic, dtype=bool)
    plastic = pl[ok] if pl.shape == ok.shape else np.zeros(int(ok.sum()), dtype=bool)
    if rng is not None:
        fresh = np.ones(n_new, dtype=bool)
        fresh[mapping[mapping >= 0]] = False  # neurons that existed before
        new_in = np.flatnonzero(fresh[:n_in_new])
        new_out = n_in_new + np.flatnonzero(fresh[n_in_new : n_in_new + new_outputs])
        targets = np.arange(n_in_new, n_new)  # every non-input neuron
        sources = np.arange(n_new)
        pairs = [(s_, t) for s_ in new_in for t in targets]  # new senses -> brain
        pairs += [(s_, t) for t in new_out for s_ in sources if s_ != t]  # brain -> new muscles
        if pairs:
            ps = np.array(pairs, dtype=np.int64).reshape(-1, 2)
            keep = rng.random(len(ps)) < spec.init_connection_prob
            ps = ps[keep]
            src = np.concatenate([src, ps[:, 0]])
            dst = np.concatenate([dst, ps[:, 1]])
            weight = np.concatenate([weight, rng.normal(0.0, spec.init_weight_sigma, len(ps))])
            delay = np.concatenate([delay, rng.integers(1, spec.max_delay + 1, len(ps))])
            plastic = np.concatenate([plastic, np.full(len(ps), spec.plastic)])
    adapted = Genome(
        n_inputs=n_in_new,
        n_outputs=new_outputs,
        tau_m=params(genome.tau_m, spec.tau_m),
        v_thresh=params(genome.v_thresh, spec.v_thresh),
        v_reset=params(genome.v_reset, spec.v_reset),
        refractory_ms=params(genome.refractory_ms, spec.refractory_ms),
        dormant=params(genome.dormant.astype(float), 0.0) > 0.5,
        wake_at=params(genome.wake_at, spec.init_wake_at),
        src=src,
        dst=dst,
        weight=weight,
        delay=delay,
        plastic=plastic,
        learning_rate=genome.learning_rate,
    )
    return adapted.repair(spec)
