import numpy as np
import pytest

from spiking_evo.genetics.population import Evolution, merge_evaluations


def assert_same(a, b):
    assert np.array_equal(a.fitness, b.fitness)
    for t in a.terms:
        assert np.array_equal(a.terms[t], b.terms[t]), t
    for f in a.stats.__dataclass_fields__:
        assert np.array_equal(getattr(a.stats, f), getattr(b.stats, f)), f


def random_everywhere(tiny_config, encoder):
    # Every per-agent random source on, and agents that die at different times.
    return tiny_config(
        world={"same_start": False},
        agent={"initial_energy": 2.0},
        stimuli=[
            {"type": "food", "count": 12, "regrow_rate": 0.1, "shared_layout": False},
            {"type": "hazard", "count": 2, "radius": 1.5, "shared_layout": False},
        ],
        network={"encoder": encoder},
        fitness={"episodes": 2},
    )


@pytest.mark.parametrize("encoder", ["rate", "latency", "current"])
@pytest.mark.parametrize("bounds", [(0, 5, 12), (0, 1, 2, 3, 12), (0, 11, 12)])
def test_chunked_evaluation_is_identical(tiny_config, encoder, bounds):
    evo = Evolution(random_everywhere(tiny_config, encoder))
    pop = evo.initial_population()
    full = evo.evaluate(pop, 3)
    assert (full.stats.final_energy == 0).any() and (full.stats.final_energy > 0).any()
    parts = [
        evo.evaluate_chunk(pop[lo:hi], 3, lo, len(pop))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    assert_same(full, merge_evaluations(parts))


def test_worker_pool_run_matches_serial(tiny_config, tmp_path):
    cfg = random_everywhere(tiny_config, "rate")
    serial = Evolution(cfg).run(out_root=tmp_path / "serial")
    parallel = Evolution(cfg, workers=3).run(out_root=tmp_path / "parallel")
    assert all(a.equals(b) for a, b in zip(serial, parallel, strict=True))


def test_shared_arenas_cannot_be_split(tiny_config):
    with pytest.raises(ValueError, match="isolated"):
        Evolution(tiny_config(world={"arenas": "shared"}), workers=2)


def test_blinding_a_stimulus_zeroes_only_its_senses(tiny_config):
    cfg = tiny_config(stimuli=[{"type": "food", "count": 12}, {"type": "hazard", "count": 2}])
    pop = Evolution(cfg).initial_population()
    seen = Evolution(cfg).evaluate(pop, 0)
    blind = Evolution(cfg, blind=("food",)).evaluate(pop, 0)
    assert not np.array_equal(seen.fitness, blind.fitness)
    assert np.array_equal(
        Evolution(cfg, blind=("food",), workers=2).evaluate(pop, 0).fitness, blind.fitness
    )


@pytest.mark.parametrize("bounds", [(0, 5, 12), (0, 11, 12)])
def test_chunked_evaluation_with_patches_and_others(tiny_config, bounds):
    # New E0 stimuli in isolated arenas: results must not depend on the split.
    cfg = tiny_config(
        world={"same_start": False},
        stimuli=[
            {
                "type": "food_patch",
                "count": 12,
                "patches": 2,
                "regrow_rate": 0.1,
                "relocate_every": 20,
                "shared_layout": False,
            },
            {"type": "others"},
        ],
        fitness={"episodes": 2},
    )
    evo = Evolution(cfg)
    pop = evo.initial_population()
    full = evo.evaluate(pop, 3)
    parts = [
        evo.evaluate_chunk(pop[lo:hi], 3, lo, len(pop))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    assert_same(full, merge_evaluations(parts))
