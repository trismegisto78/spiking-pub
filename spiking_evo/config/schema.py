"""Declarative experiment configuration.

Every tunable number of the simulation lives here, with a default. Experiment files
(YAML or TOML) override any subset of these fields; unknown keys are an error so that
typos never silently fall back to defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldConfig:
    width: int = 40
    height: int = 40
    steps_per_episode: int = 1000
    # Toroidal grid when True; otherwise the borders are walls.
    wrap: bool = True
    # "isolated": every agent lives in its own copy of the world (no competition).
    # "shared": all agents live in the same world and compete for resources.
    arenas: str = "isolated"
    # All agents start from the same position/heading (fair comparison) when True.
    same_start: bool = True


@dataclass
class AgentConfig:
    initial_energy: float = 50.0
    metabolic_cost: float = 0.05  # energy lost every world step while alive
    move_cost: float = 0.02  # extra energy lost for each cell moved
    turn_cost: float = 0.0  # extra energy lost for each 90 degree turn
    die_at_zero: bool = True  # dead agents stop moving and interacting


@dataclass
class NeuronConfig:
    """Initial LIF parameters. They are stored per neuron in the genome."""

    tau_m: float = 20.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    refractory_ms: float = 2.0


@dataclass
class PlasticityConfig:
    """Reward-modulated STDP: synapses learn during the agent's life (see snn/network.py).

    Off by default. When on, every genome carries a learning rate and a per-synapse
    "plastic" flag, both evolvable; weights are reset to the genome's at every birth."""

    enabled: bool = False
    tau_trace_ms: float = 20.0  # pre/post spike traces of the STDP window
    tau_eligibility_ms: float = 500.0  # how long a spike pairing stays eligible for reward
    a_minus_ratio: float = 1.05  # depression amplitude relative to potentiation
    # Reward signal: "energy" = total energy change of the step (includes the constant cost
    # of living and moving); "stimuli" = only what stimuli gave or took (food, crops,
    # hazards, poison...), a cleaner signal.
    reward: str = "energy"
    reward_scale: float = 5.0  # energy change counted as a reward of 1
    reward_baseline_steps: float = 50.0  # running average the reward is compared with
    init_learning_rate: float = 0.05  # learning rate of the initial population
    learning_rate_max: float = 1.0
    init_plastic_prob: float = 0.5  # share of plastic synapses in the initial population


@dataclass
class DevelopmentConfig:
    """Brain development: hidden neurons may be dormant at birth and wake up during life
    when the agent has met enough "difficulty" (energy lost to stimuli: damage, poison,
    costs). Each awake non-input neuron costs ``neuron_cost`` energy per world step. The
    dormant flag and the wake-up threshold of every neuron are genes."""

    enabled: bool = False
    init_dormant_prob: float = 0.5  # share of hidden neurons dormant at birth (initial pop)
    init_wake_at: float = 20.0  # difficulty needed to wake a dormant neuron (initial pop)
    neuron_cost: float = 0.002  # energy per awake non-input neuron per world step
    plastic_new_only: bool = False  # with plasticity: only synapses of woken neurons learn
    # What wakes dormant neurons: "difficulty" (energy taken by stimuli) or "growth" (energy
    # the agent invests in its brain; used by the ecosystem, see ecosystem.growth_rate).
    trigger: str = "difficulty"


@dataclass
class NetworkConfig:
    # Sensor encoding: "rate" (Bernoulli spikes), "latency" (time-to-first-spike
    # inside each tick) or "current" (intensity injected as constant current).
    encoder: str = "rate"
    hidden_max: int = 30
    hidden_init: int = 5
    dt_ms: float = 1.0
    # SNN integration steps per world step; the motor window is this long.
    steps_per_tick: int = 10
    max_delay: int = 5  # synaptic delays are integers in [1, max_delay] SNN steps
    bias_input: bool = True  # extra input channel with constant intensity 1
    max_rate_hz: float = 200.0  # rate encoder: firing rate at intensity 1
    current_gain: float = 2.0  # current encoder: injected current at intensity 1
    # Motor decoding: act when the rate difference of an antagonist pair exceeds this.
    motor_threshold_hz: float = 0.0
    # Extra "action" output neurons, beyond the 4 movement neurons. What an action does is
    # decided by the stimuli that read ``agents.action``; the core only delivers it.
    action_outputs: int = 0
    action_threshold_hz: float = 50.0  # an action fires when its neuron exceeds this rate
    init_connection_prob: float = 0.3
    init_weight_sigma: float = 1.0
    weight_max: float = 5.0
    thresh_range: tuple[float, float] = (0.1, 5.0)
    tau_range: tuple[float, float] = (2.0, 100.0)
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)


@dataclass
class MutationConfig:
    # Per-gene probabilities.
    weight: float = 0.1
    weight_sigma: float = 0.3
    threshold: float = 0.05
    threshold_sigma: float = 0.1
    tau: float = 0.0
    tau_sigma: float = 0.1
    delay: float = 0.01
    # Per-genome probabilities of a structural event.
    add_synapse: float = 0.02
    remove_synapse: float = 0.01
    add_neuron: float = 0.01
    remove_neuron: float = 0.005
    # Plasticity genes (used only when network.plasticity.enabled).
    learning_rate: float = 0.1  # per-genome probability of a learning-rate change
    learning_rate_sigma: float = 0.02
    plastic_flip: float = 0.01  # per-synapse probability of switching plastic on/off
    # Development genes (used only when network.development.enabled).
    dormant_flip: float = 0.02  # per-neuron probability of switching dormant on/off
    wake_at: float = 0.1  # per-neuron probability of a wake-threshold change
    wake_at_sigma: float = 0.3  # log-normal step of that change


@dataclass
class GeneticsConfig:
    population: int = 200
    generations: int = 300
    tournament_size: int = 3
    elitism: int = 5
    crossover: float = 0.0  # probability that an offspring comes from crossover
    encoding: str = "direct"  # "indirect" is planned for milestone 3
    mutation: MutationConfig = field(default_factory=MutationConfig)


@dataclass
class FiringPenaltyConfig:
    target_rate_hz: float = 20.0
    weight: float = 0.5


@dataclass
class WeightOnlyConfig:
    weight: float = 0.0


@dataclass
class FitnessConfig:
    energy: float = 1.0  # weight of the energy gained from stimuli
    firing_penalty: FiringPenaltyConfig = field(default_factory=FiringPenaltyConfig)
    silence_penalty: WeightOnlyConfig = field(default_factory=lambda: WeightOnlyConfig(0.5))
    exploration: WeightOnlyConfig = field(default_factory=WeightOnlyConfig)
    episodes: int = 1  # episodes per evaluation; terms are averaged


@dataclass
class MetricsConfig:
    out_dir: str = "runs"
    snapshot_every: int = 10  # best genome snapshot every N generations (0 = only last)
    checkpoint_every: int = 10  # full population checkpoint for --resume (0 = only last)


@dataclass
class EcosystemConfig:
    """Continuous, shared-world evolution (stage D): births, deaths, no fitness function.
    Used by `spiking-evo ecosystem`; ignored by the generational runs."""

    capacity: int = 300  # maximum number of living agents (slots)
    # Start from an existing species: a run directory or checkpoint (.json/.json.gz) whose
    # genomes seed the initial population (adapted to this world's senses). Empty = random.
    init_from: str = ""
    initial_population: int = 120
    ticks: int = 50000
    log_every: int = 250
    snapshot_every: int = 10000  # 0 = only at the end
    max_age: int = 3000  # agents die of old age
    juvenile_speed: float = 0.5  # share of steps on which a juvenile may move
    juvenile_damage: float = 2.0  # multiplier of the energy juveniles lose to stimuli
    juvenile_can_act: bool = False  # may juveniles use the action outputs (e.g. farm)
    repro_overhead: float = 0.1  # share of the endowment lost in reproduction
    repro_cooldown: int = 30  # minimum steps between two reproductions of an agent
    spawn_radius: int = 1  # offspring appear within this distance of the parent
    # Life-history genes of the initial population (they evolve, within the bounds).
    init_repro_threshold: float = 100.0  # energy needed to reproduce
    init_endowment: float = 0.5  # share of the parent's energy given to the offspring
    init_clutch: int = 2  # offspring per reproduction
    init_juvenile_age: float = 150.0  # steps spent as a juvenile
    repro_threshold_range: tuple[float, float] = (20.0, 500.0)
    endowment_range: tuple[float, float] = (0.05, 0.9)
    clutch_max: int = 6
    juvenile_age_range: tuple[float, float] = (0.0, 1000.0)
    trait_sigma: float = 0.1  # log-normal mutation step of the continuous traits
    # Brain growth (with network.development.trigger "growth"): every step an agent with
    # dormant neurons invests up to growth_rate energy in its brain, keeping at least
    # growth_reserve; a dormant neuron wakes when the invested energy reaches its wake_at.
    growth_rate: float = 0.1
    growth_reserve: float = 10.0
    clutch_change: float = 0.1  # probability that clutch size changes by +-1 at birth


@dataclass
class Config:
    experiment: str = "unnamed"
    seed: int = 0
    world: WorldConfig = field(default_factory=WorldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # Each entry: {"type": <registered name>, ...stimulus parameters}.
    stimuli: list[dict[str, Any]] = field(default_factory=lambda: [{"type": "food"}])
    network: NetworkConfig = field(default_factory=NetworkConfig)
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # Environment changes during a run. Each phase: {from_generation, name, set}; ``set`` is
    # merged into the previous phase's config from that generation on (phases accumulate;
    # lists such as ``stimuli`` are replaced as a whole). The population carries over:
    # genomes are adapted to new senses.
    phases: list[dict[str, Any]] = field(default_factory=list)
    ecosystem: EcosystemConfig = field(default_factory=EcosystemConfig)
